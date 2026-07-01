from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
from openai import OpenAI
from gtts import gTTS
from moviepy import ColorClip, AudioFileClip
import imageio_ffmpeg
import asyncio
import time
import uuid
import json
from pathlib import Path

os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

JOBS_DIR = Path("/tmp/jobs")
JOBS_DIR.mkdir(exist_ok=True)

def save_job(job_id, job_data):
    with open(JOBS_DIR / f"{job_id}.json", "w") as f:
        json.dump(job_data, f)

def load_job(job_id):
    job_file = JOBS_DIR / f"{job_id}.json"
    if not job_file.exists():
        return None
    with open(job_file, "r") as f:
        return json.load(f)

def update_job(job_id, updates):
    job = load_job(job_id)
    if job:
        job.update(updates)
        save_job(job_id, job)

@app.get("/")
def root():
    return {"status": "ViralForge API v3"}

@app.post("/generate-video")
async def generate_video(
    topic: str = Query(...),
    duration: int = Query(default=10, ge=5, le=15)
):
    job_id = str(uuid.uuid4())
    job_data = {
        "job_id": job_id, 
        "status": "processing", 
        "progress": 0,
        "topic": topic, 
        "duration": duration, 
        "created_at": time.time(), 
        "video_path": None
    }
    save_job(job_id, job_data)
    asyncio.create_task(create_video(job_id, topic, duration))
    return {"job_id": job_id, "status": "processing"}

@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    job = load_job(job_id)
    if not job: 
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id, 
        "status": job["status"], 
        "progress": job.get("progress", 0),
        "error": job.get("error")
    }

@app.get("/download-video/{job_id}")
async def download_video(job_id: str):
    job = load_job(job_id)
    if not job: 
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed": 
        raise HTTPException(status_code=400, detail="Video not ready yet")
    return FileResponse(
        job["video_path"], 
        media_type="video/mp4", 
        filename=f"viralforge_{job['topic'].replace(' ', '_')}.mp4"
    )

async def create_video(job_id: str, topic: str, duration: int):
    try:
        print(f"[{job_id}] Starting...")
        
        # 20% - Script
        update_job(job_id, {"progress": 20})
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Write a {duration}-second script about {topic}. {duration*2} words max."}]
        )
        script = response.choices[0].message.content
        
        # 40% - Audio
        update_job(job_id, {"progress": 40})
        tts = gTTS(text=script, lang='en', slow=False)
        audio_path = f"/tmp/audio_{job_id}.mp3"
        tts.save(audio_path)
        audio = AudioFileClip(audio_path)
        audio_duration = audio.duration
        
        # 70% - Create clip
        update_job(job_id, {"progress": 70})
        
        # ✅ CORRECT MoviePy 2.x syntax:
        clip = ColorClip(
            size=(480, 854), 
            color=(139, 92, 246),
            duration=audio_duration
            # NO fps here!
        )
        
        # Set FPS separately
        clip = clip.with_fps(15)
        
        # Add audio
        clip = clip.with_audio(audio)
        
        # 90% - Render
        update_job(job_id, {"progress": 90})
        
        output_path = f"/tmp/video_{job_id}.mp4"
        clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            preset="ultrafast",
            bitrate="500k",
            threads=1,
            verbose=False,
            logger=None
        )
        
        audio.close()
        clip.close()
        
        # 100% - Done
        update_job(job_id, {
            "status": "completed", 
            "progress": 100, 
            "video_path": output_path
        })
        
        print(f"[{job_id}] SUCCESS!")
        
    except Exception as e:
        error_msg = str(e)
        print(f"[{job_id}] FAILED: {error_msg}")
        update_job(job_id, {
            "status": "failed", 
            "error": error_msg
        })

# Cleanup
import shutil
if JOBS_DIR.exists():
    shutil.rmtree(JOBS_DIR)
    JOBS_DIR.mkdir(exist_ok=True)
