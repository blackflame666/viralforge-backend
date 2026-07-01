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

# Store job status (in production, use Redis/database)
jobs = {}

@app.get("/")
def root():
    return {"status": "ViralForge API - Pro Plan Active"}

@app.post("/generate-video")
async def generate_video(
    topic: str = Query(..., description="Video topic"),
    duration: int = Query(default=15, ge=10, le=30)
):
    """Start a video generation job (returns immediately)"""
    job_id = str(uuid.uuid4())
    
    jobs[job_id] = {
        "status": "processing",
        "topic": topic,
        "duration": duration,
        "created_at": time.time(),
        "video_path": None
    }
    
    # Start video generation in background
    asyncio.create_task(create_video(job_id, topic, duration))
    
    return {
        "job_id": job_id,
        "status": "processing",
        "message": "Video generation started. Use job_id to check status."
    }

@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    """Check if video is ready"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job.get("progress", 0)
    }

@app.get("/download-video/{job_id}")
async def download_video(job_id: str):
    """Download the finished video"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Video not ready yet")
    
    return FileResponse(
        job["video_path"],
        media_type="video/mp4",
        filename=f"viralforge_{job['topic'].replace(' ', '_')}.mp4"
    )

async def create_video(job_id: str, topic: str, duration: int):
    """Background task to create the video"""
    try:
        job = jobs[job_id]
        
        # Step 1: Generate script (20%)
        job["progress"] = 20
        word_count = min(duration * 2, 40)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Write a {duration}-second viral script about {topic}. Use {word_count} words max."}]
        )
        script = response.choices[0].message.content
        
        # Step 2: Generate voiceover (40%)
        job["progress"] = 40
        tts = gTTS(text=script, lang='en', slow=False)
        audio_path = f"audio_{job_id}.mp3"
        tts.save(audio_path)
        audio = AudioFileClip(audio_path)
        
        # Step 3: Get stock footage (60%)
        job["progress"] = 60
        headers = {"Authorization": PEXELS_API_KEY}
        search_url = f"https://api.pexels.com/videos/search?query={topic}&per_page=2&orientation=portrait"
        pexels_res = requests.get(search_url, headers=headers, timeout=10).json()
        
        video_clips = []
        clip_duration = audio.duration / 2
        
        for i, video in enumerate(pexels_res['videos'][:2]):
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
        
        # Step 4: Combine and render (80%)
        job["progress"] = 80
        final_video = concatenate_videoclips(video_clips, method="compose")
        final_video = final_video.with_audio(audio)
        
        output_path = f"video_{job_id}.mp4"
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
        
        # Clean up audio
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        # Complete (100%)
        job["status"] = "completed"
        job["progress"] = 100
        job["video_path"] = output_path
        
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        print(f"Job {job_id} failed: {str(e)}")

# Cleanup old jobs every hour
async def cleanup_old_jobs():
    while True:
        await asyncio.sleep(3600)
        current_time = time.time()
        expired_jobs = [jid for jid, job in jobs.items() 
                       if current_time - job["created_at"] > 3600]
        for jid in expired_jobs:
            if jobs[jid].get("video_path") and os.path.exists(jobs[jid]["video_path"]):
                os.remove(jobs[jid]["video_path"])
            del jobs[jid]

# Start cleanup task
asyncio.create_task(cleanup_old_jobs())
