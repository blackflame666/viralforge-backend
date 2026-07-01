from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
from openai import OpenAI
from gtts import gTTS
from moviepy.editor import ColorClip, AudioFileClip
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
    return {"status": "ViralForge API - Ultra Lite"}

@app.post("/generate-video")
async def generate_video(
    topic: str = Query(...),
    duration: int = Query(default=10, ge=5, le=15)
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
        print(f"[{job_id}] Starting Ultra Lite generation...")
        
        # Step 1: Generate script (20%)
        update_job(job_id, {"progress": 20})
        print(f"[{job_id}] Generating script...")
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Write a {duration}-second viral script about {topic}. Keep it under {duration*2} words."}]
        )
        script = response.choices[0].message.content
        print(f"[{job_id}] Script: {script[:50]}...")
        
        # Step 2: Generate audio (40%)
        update_job(job_id, {"progress": 40})
        print(f"[{job_id}] Generating audio...")
        
        tts = gTTS(text=script, lang='en', slow=False)
        audio_path = f"/tmp/audio_{job_id}.mp3"
        tts.save(audio_path)
        
        # Get audio duration
        audio = AudioFileClip(audio_path)
        audio_duration = audio.duration
        print(f"[{job_id}] Audio duration: {audio_duration:.2f}s")
        
        # Step 3: Create simple video (70%)
        update_job(job_id, {"progress": 70})
        print(f"[{job_id}] Creating video with gradient background...")
        
        # Create a single gradient clip (purple to blue)
        # This is MUCH simpler than concatenating
        def make_frame(t):
            # Simple gradient animation
            if t < audio_duration / 2:
                return [[139, 92, 246] for _ in range(480 * 854)]  # Purple
            else:
                return [[59, 130, 246] for _ in range(480 * 854)]  # Blue
        
        # Even simpler: Just use one solid color clip
        clip = ColorClip(size=(480, 854), color=(139, 92, 246), duration=audio_duration)
        clip = clip.set_fps(15)
        clip = clip.set_audio(audio)
        
        # Step 4: Render (90%)
        update_job(job_id, {"progress": 90})
        print(f"[{job_id}] Rendering video (this may take a minute)...")
        
        output_path = f"/tmp/video_{job_id}.mp4"
        
        clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=15,
            preset="ultrafast",
            bitrate="500k",
            threads=1,
            verbose=False,
            logger=None
        )
        
        # Cleanup
        audio.close()
        clip.close()
        
        # Step 5: Complete (100%)
        update_job(job_id, {
            "status": "completed", 
            "progress": 100, 
            "video_path": output_path
        })
        
        print(f"[{job_id}] ✅ SUCCESS! Video saved to {output_path}")
        
    except Exception as e:
        error_msg = f"{str(e)}"
        print(f"[{job_id}] ❌ FAILED: {error_msg}")
        import traceback
        traceback.print_exc()
        
        update_job(job_id, {
            "status": "failed", 
            "error": error_msg
        })

# Cleanup old jobs on startup
import shutil
if JOBS_DIR.exists():
    shutil.rmtree(JOBS_DIR)
    JOBS_DIR.mkdir(exist_ok=True)
