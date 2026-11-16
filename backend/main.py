import os
import json
import uuid
import sqlite3
import subprocess
from datetime import datetime, timedelta
from typing import Literal, Optional

import razorpay
import firebase_admin
from firebase_admin import auth as firebase_auth, credentials
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from faster_whisper import WhisperModel
from google import genai
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="ClipForge AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("clips", exist_ok=True)
app.mount("/clips", StaticFiles(directory="clips"), name="clips")

if not firebase_admin._apps:
    sa_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    if sa_json:
        cred = credentials.Certificate(json.loads(sa_json))
    else:
        cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

rzp_client = razorpay.Client(
    auth=(os.getenv("RAZORPAY_KEY_ID"), os.getenv("RAZORPAY_KEY_SECRET"))
)

conn = sqlite3.connect("users.db", check_same_thread=False)
conn.execute("""
CREATE TABLE IF NOT EXISTS users (
    uid TEXT PRIMARY KEY,
    email TEXT,
    clips_used INTEGER DEFAULT 0,
    is_pro INTEGER DEFAULT 0,
    pro_expires_at TEXT
)
""")
conn.commit()


def get_current_user(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid token header")
    token = authorization.split("Bearer ")[1]
    try:
        decoded = firebase_auth.verify_id_token(token)
        uid = decoded["uid"]
        email = decoded.get("email", "")
        cur = conn.cursor()
        cur.execute("INSERT OR IGNORE INTO users (uid, email) VALUES (?, ?)", (uid, email))
        conn.commit()
        return {"uid": uid, "email": email}
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid Google login token")


def check_user_plan(uid: str):
    cur = conn.cursor()
    cur.execute("SELECT clips_used, is_pro, pro_expires_at FROM users WHERE uid = ?", (uid,))
    clips_used, is_pro, expires_at = cur.fetchone()
    if is_pro and expires_at and datetime.fromisoformat(expires_at) < datetime.utcnow():
        is_pro = 0
        cur.execute("UPDATE users SET is_pro = 0 WHERE uid = ?", (uid,))
        conn.commit()
    return {"clips_used": clips_used, "is_pro": bool(is_pro), "pro_expires_at": expires_at}


class ClipRequest(BaseModel):
    url: str
    duration: Literal[30, 45, 60] = 30
    resolution: Literal["720p", "1080p", "4K"] = "1080p"
    max_clips: int = 3


class OrderRequest(BaseModel):
    plan_id: Literal["INR_1M", "INR_3M", "USD_1M", "USD_3M"]


class VerifyPaymentRequest(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str
    plan_id: Literal["INR_1M", "INR_3M", "USD_1M", "USD_3M"]


@app.get("/api/me")
def get_profile(
    user=Depends(get_current_user),
    x_vercel_ip_country: Optional[str] = Header(None),
    cf_ipcountry: Optional[str] = Header(None),
):
    plan = check_user_plan(user["uid"])
    country = (x_vercel_ip_country or cf_ipcountry or "IN").upper()
    is_india = country == "IN"
    return {
        **user,
        **plan,
        "region": "IN" if is_india else "GLOBAL",
        "pricing": {
            "currency": "INR" if is_india else "USD",
            "symbol": "₹" if is_india else "$",
            "monthly": 20 if is_india else 1,
            "quarterly": 50 if is_india else 3,
            "monthly_plan_id": "INR_1M" if is_india else "USD_1M",
            "quarterly_plan_id": "INR_3M" if is_india else "USD_3M",
        },
    }


@app.post("/api/create-order")
def create_payment_order(req: OrderRequest, user=Depends(get_current_user)):
    plans = {
        "INR_1M": {"amount": 2000, "currency": "INR"},
        "INR_3M": {"amount": 5000, "currency": "INR"},
        "USD_1M": {"amount": 100, "currency": "USD"},
        "USD_3M": {"amount": 300, "currency": "USD"},
    }
    selected = plans[req.plan_id]
    order = rzp_client.order.create({
        "amount": selected["amount"],
        "currency": selected["currency"],
        "payment_capture": 1,
        "notes": {"uid": user["uid"], "plan_id": req.plan_id},
    })
    return {
        "order_id": order["id"],
        "amount": selected["amount"],
        "currency": selected["currency"],
        "key_id": os.getenv("RAZORPAY_KEY_ID"),
    }


@app.post("/api/verify-payment")
def verify_payment(req: VerifyPaymentRequest, user=Depends(get_current_user)):
    try:
        rzp_client.utility.verify_payment_signature({
            "razorpay_order_id": req.razorpay_order_id,
            "razorpay_payment_id": req.razorpay_payment_id,
            "razorpay_signature": req.razorpay_signature,
        })
    except Exception:
        raise HTTPException(status_code=400, detail="Payment verification failed")

    days = 90 if req.plan_id.endswith("3M") else 30
    expiry = (datetime.utcnow() + timedelta(days=days)).isoformat()
    cur = conn.cursor()
    cur.execute(
        "UPDATE users SET is_pro = 1, pro_expires_at = ? WHERE uid = ?",
        (expiry, user["uid"]),
    )
    conn.commit()
    return {"status": "success", "pro_expires_at": expiry}


@app.post("/api/generate-clips")
def generate_clips(req: ClipRequest, user=Depends(get_current_user)):
    plan = check_user_plan(user["uid"])

    if not plan["is_pro"]:
        if req.resolution == "4K":
            raise HTTPException(status_code=402, detail="4K requires Pro.")
        if plan["clips_used"] + req.max_clips > 15:
            raise HTTPException(status_code=402, detail="Free limit of 15 clips reached.")

    job_id = str(uuid.uuid4())[:8]
    audio_path = f"clips/{job_id}_audio.m4a"

    try:
        subprocess.run(
            ["yt-dlp", "-f", "bestaudio[ext=m4a]/bestaudio", "-o", audio_path, req.url],
            check=True,
        )

        segments, _ = whisper_model.transcribe(audio_path, beam_size=1)
        transcript_lines = [
            f"[{seg.start:.1f}s - {seg.end:.1f}s]: {seg.text.strip()}"
            for seg in segments
        ]
        full_transcript = "\n".join(transcript_lines[:500])

        prompt = f"""
        Pick the top {req.max_clips} viral YouTube Shorts / Instagram Reels moments from this transcript.
        Each clip MUST be approximately {req.duration} seconds long.
        Return ONLY valid JSON list with keys:
        "start" (float seconds), "end" (float seconds), "title" (string), "viral_score" (1-100).

        Transcript:
        {full_transcript}
        """
        ai_res = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        raw_json = ai_res.text.strip().removeprefix("```json").removesuffix("```").strip()
        highlights = json.loads(raw_json)[:req.max_clips]

        height_map = {"720p": 720, "1080p": 1080, "4K": 2160}
        max_h = height_map[req.resolution]
        format_str = f"bestvideo[height<={max_h}]+bestaudio/best[height<={max_h}]"

        output_clips = []
        for idx, item in enumerate(highlights):
            start_sec = max(0, float(item["start"]))
            end_sec = start_sec + req.duration
            out_file = f"clips/{job_id}_clip_{idx+1}_{req.resolution}.mp4"

            subprocess.run(
                [
                    "yt-dlp",
                    "-f", format_str,
                    "--download-sections", f"*{start_sec}-{end_sec}",
                    "--merge-output-format", "mp4",
                    "-o", out_file,
                    req.url,
                ],
                check=True,
            )
            output_clips.append({
                "title": item.get("title", f"Viral Clip #{idx+1}"),
                "viral_score": item.get("viral_score", 90),
                "duration": req.duration,
                "resolution": req.resolution,
                "download_url": f"/{out_file}",
            })

        cur = conn.cursor()
        cur.execute(
            "UPDATE users SET clips_used = clips_used + ? WHERE uid = ?",
            (len(output_clips), user["uid"]),
        )
        conn.commit()

        return {"clips": output_clips, "clips_used": plan["clips_used"] + len(output_clips)}

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
