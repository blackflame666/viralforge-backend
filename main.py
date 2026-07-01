from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import requests
import tempfile
from openai import OpenAI
from gtts import gTTS
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips, ColorClip
import imageio_ffmpeg
import asyncio
import time
import uuid
import json
import gc  # Garbage collection
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
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY")
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
    return {"status": "ViralForge API - Lite Mode"}

@app.post("/generate-video")
async def generate_video(
    topic: str = Query(...),
    duration: int = Query(default=10, ge=5, le=15)  # SHORTER videos
):
    job_id = str(uuid.uuid4())
    job_data = {
        "job_id": job_id, "status": "processing", "progress": 0,
        "topic": topic, "duration": duration, "created_at": time.time(), "video_path": None
    }
    save_job(job_id, job_data)
    asyncio.create_task(create_video(job_id, topic, duration))
    return {"job_id": job_id, "status": "processing"}

@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    job = load_job(job_id)
    if not job: raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "status": job["status"], "progress": job.get("progress", 0)}

@app.get("/download-video/{job_id}")
async def download_video(job_id: str):
    job = load_job(job_id)
    if not job: raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed": raise HTTPException(status_code=400, detail="Video not ready")
    return FileResponse(job["video_path"], media_type="video/mp4", filename=f"viralforge_{job['topic'].replace(' ', '_')}.mp4")

async def create_video(job_id: str, topic: str, duration: int):
    try:
        print(f"[{job_id}] Starting (Lite Mode)...")
        update_job(job_id, {"progress": 10})
        
        # 1. Script
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Write a {duration}-second script about {topic}. {duration*2} words max."}]
        )
        script = response.choices[0].message.content
        update_job(job_id, {"progress": 30})
        
        # 2. Audio
        tts = gTTS(text=script, lang='en', slow=False)
        audio_path = f"/tmp/audio_{job_id}.mp3"
        tts.save(audio_path)
        audio = AudioFileClip(audio_path)
        audio_duration = audio.duration
        update_job(job_id, {"progress": 50})
        
        # 3. SKIP PEXELS - Just use colored clips (MUCH lighter on memory)
        print(f"[{job_id}] Creating gradient background...")
        update_job(job_id, {"progress": 60})
        
        # Create simple gradient clips (no downloads = no memory issues)
        clip_duration = audio_duration / 2
        
        # Clip 1: Purple gradient
        clip1 = ColorClip(size=(480, 854), color=(139, 92, 246), duration=clip_duration)  # Purple
        # Clip 2: Blue gradient  
        clip2 = ColorClip(size=(480, 854), color=(59, 130, 246), duration=clip_duration)  # Blue
        
        video_clips = [clip1, clip2]
        
        update_job(job_id, {"progress": 70})
        
        # 4. Combine (LOW memory settings)
        print(f"[{job_id}] Rendering video...")
        final_video = concatenate_videoclips(video_clips, method="compose")
        final_video = final_video.with_audio(audio)
        
        output_path = f"/tmp/video_{job_id}.mp4"
        
        # VERY low memory settings
        final_video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=15,  # Lower FPS
            preset="ultrafast",
            bitrate="800k",  # Lower bitrate
            threads=1,  # Single thread = less memory
            verbose=False,
            logger=None,
            temp_audiofile="/tmp/audio_temp.wav"
        )
        
        # Force garbage collection
        gc.collect()
        
        print(f"[{job_id}] ✅ Complete!")
        update_job(job_id, {"status": "completed", "progress": 100, "video_path": output_path})
        
        # Cleanup
        if os.path.exists(audio_path):
            os.remove(audio_path)
        audio.close()
        
    except Exception as e:
        print(f"[{job_id}] ❌ Failed: {str(e)}")
        update_job(job_id, {"status": "failed", "error": str(e)})
