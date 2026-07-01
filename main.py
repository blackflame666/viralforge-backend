from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import os
import requests
import tempfile
from openai import OpenAI
from gtts import gTTS
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips
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

# Create jobs directory
JOBS_DIR = Path("/tmp/jobs")
JOBS_DIR.mkdir(exist_ok=True)

def save_job(job_id, job_data):
    """Save job to file"""
    with open(JOBS_DIR / f"{job_id}.json", "w") as f:
        json.dump(job_data, f)

def load_job(job_id):
    """Load job from file"""
    job_file = JOBS_DIR / f"{job_id}.json"
    if not job_file.exists():
        return None
    with open(job_file, "r") as f:
        return json.load(f)

def update_job(job_id, updates):
    """Update job fields"""
    job = load_job(job_id)
    if job:
        job.update(updates)
        save_job(job_id, job)

@app.get("/")
def root():
    return {"status": "ViralForge API - Pro Plan Active"}

@app.post("/generate-video")
async def generate_video(
    topic: str = Query(..., description="Video topic"),
    duration: int = Query(default=15, ge=10, le=30)
):
    """Start a video generation job"""
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
    
    # Start background task
    asyncio.create_task(create_video(job_id, topic, duration))
    
    return {
        "job_id": job_id,
        "status": "processing",
        "message": "Video generation started"
    }

@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    """Check job status"""
    job = load_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job.get("progress", 0)
    }

@app.get("/download-video/{job_id}")
async def download_video(job_id: str):
    """Download finished video"""
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
    """Background video creation task"""
    try:
        update_job(job_id, {"progress": 10})
        
        # Generate script
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Write a {duration}-second viral script about {topic}. {duration*2} words max."}]
        )
        script = response.choices[0].message.content
        
        update_job(job_id, {"progress": 30})
        
        # Generate audio
        tts = gTTS(text=script, lang='en', slow=False)
        audio_path = f"/tmp/audio_{job_id}.mp3"
        tts.save(audio_path)
        audio = AudioFileClip(audio_path)
        
        update_job(job_id, {"progress": 50})
        
        # Get stock footage
        headers = {"Authorization": PEXELS_API_KEY}
        search_url = f"https://api.pexels.com/videos/search?query={topic}&per_page=2&orientation=portrait"
        pexels_res = requests.get(search_url, headers=headers, timeout=10).json()
        
        video_clips = []
        clip_duration = audio.duration / 2
        
        for video in pexels_res['videos'][:2]:
            video_url = video['video_files'][0]['link']
            vid_res = requests.get(video_url, timeout=15)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
                tmp_vid.write(vid_res.content)
                clip = VideoFileClip(tmp_vid.name)
                clip = clip.resized((1080, 1920))
                
                if clip.duration < clip_duration:
                    clip = clip.loop(duration=clip_duration)
                else:
                    clip = clip.subclipped(0, clip_duration)
                
                video_clips.append(clip)
        
        update_job(job_id, {"progress": 80})
        
        # Render video
        final_video = concatenate_videoclips(video_clips, method="compose")
        final_video = final_video.with_audio(audio)
        
        output_path = f"/tmp/video_{job_id}.mp4"
        final_video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=24,
            preset="ultrafast",
            bitrate="2000k",
            verbose=False,
            logger=None
        )
        
        update_job(job_id, {
            "status": "completed",
            "progress": 100,
            "video_path": output_path
        })
        
        # Cleanup audio
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
    except Exception as e:
        update_job(job_id, {
            "status": "failed",
            "error": str(e)
        })
        print(f"Job {job_id} failed: {str(e)}")
