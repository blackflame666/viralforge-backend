from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import requests
import tempfile
from openai import OpenAI
from gtts import gTTS
from moviepy import ColorClip, AudioFileClip, VideoFileClip
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
    return {"status": "ViralForge API v4 - With Stock Footage!"}

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
    stock_video_path = None
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
        
        # 60% - Fetch Stock Footage (Safe Mode)
        update_job(job_id, {"progress": 60})
        print(f"[{job_id}] Trying to fetch stock footage for: {topic}")
        
        if PEXELS_API_KEY:
            try:
                headers = {"Authorization": PEXELS_API_KEY}
                # Search for 1 vertical video
                url = f"https://api.pexels.com/videos/search?query={topic}&per_page=1&orientation=portrait"
                res = requests.get(url, headers=headers, timeout=10).json()
                
                if res.get('videos') and len(res['videos']) > 0:
                    video_url = res['videos'][0]['video_files'][0]['link']
                    print(f"[{job_id}] Downloading stock video...")
                    vid_data = requests.get(video_url, timeout=15).content
                    
                    stock_video_path = f"/tmp/stock_{job_id}.mp4"
                    with open(stock_video_path, 'wb') as f:
                        f.write(vid_data)
                    print(f"[{job_id}] Stock video saved!")
                else:
                    print(f"[{job_id}] No videos found on Pexels.")
            except Exception as e:
                print(f"[{job_id}] Pexels error: {e}")
        else:
            print(f"[{job_id}] No Pexels API Key found in environment variables!")

        # 70% - Create Clip
        update_job(job_id, {"progress": 70})
        
        # Use stock footage if we got it, otherwise fallback to purple
        if stock_video_path and os.path.exists(stock_video_path):
            print(f"[{job_id}] Using stock footage...")
            clip = VideoFileClip(stock_video_path)
            clip = clip.resized((480, 854)) # Resize to save memory
            
            # Loop or trim to match audio
            if clip.duration < audio_duration:
                clip = clip.loop(duration=audio_duration)
            else:
                clip = clip.subclipped(0, audio_duration)
        else:
            print(f"[{job_id}] Fallback to purple screen...")
            clip = ColorClip(size=(480, 854), color=(139, 92, 246), duration=audio_duration)
            
        clip = clip.with_fps(15)
        clip = clip.with_audio(audio)
        
        # 90% - Render
        update_job(job_id, {"progress": 90})
        print(f"[{job_id}] Rendering final video...")
        
        output_path = f"/tmp/video_{job_id}.mp4"
        clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            preset="ultrafast",
            bitrate="500k",
            threads=1,
            logger=None
        )
        
        # Cleanup
        audio.close()
        clip.close()
        if stock_video_path and os.path.exists(stock_video_path):
            os.remove(stock_video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
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
