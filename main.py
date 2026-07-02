from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import requests
import tempfile
from openai import OpenAI
from gtts import gTTS
from moviepy import ColorClip, AudioFileClip, VideoFileClip, CompositeVideoClip, CompositeAudioClip, ImageClip, concatenate_videoclips
import imageio_ffmpeg
import asyncio
import time
import uuid
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

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
PIXABAY_API_KEY = os.environ.get("PIXABAY_API_KEY")

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

def create_caption_frame(text, width=680, height=150):
    try:
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
        except:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        y = (height - 40) // 2
        
        draw.text((x-2, y), text, font=font, fill='black')
        draw.text((x+2, y), text, font=font, fill='black')
        draw.text((x, y-2), text, font=font, fill='black')
        draw.text((x, y+2), text, font=font, fill='black')
        draw.text((x, y), text, font=font, fill='white')
        
        return np.array(img)
    except Exception as e:
        print(f"Caption error: {e}")
        return None

def split_text_for_captions(text, max_words=5):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = ' '.join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

@app.get("/")
def root():
    return {"status": "ViralForge API v14 - Stable!"}

@app.post("/generate-video")
async def generate_video(topic: str = Query(...), duration: int = Query(default=10, ge=5, le=15)):
    job_id = str(uuid.uuid4())
    job_data = {
        "job_id": job_id, 
        "status": "processing", 
        "progress": 10,
        "topic": topic, 
        "duration": duration, 
        "created_at": time.time(), 
        "video_path": None
    }
    save_job(job_id, job_data)
    print(f"[{job_id}] Job created: {topic}")
    asyncio.create_task(create_video(job_id, topic, duration))
    return {"job_id": job_id, "status": "processing", "progress": 10}

@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    job = load_job(job_id)
    if not job: 
        raise HTTPException(status_code=404, detail=f"Job not found")
    return {"job_id": job_id, "status": job["status"], "progress": job.get("progress", 0)}

@app.get("/download-video/{job_id}")
async def download_video(job_id: str):
    job = load_job(job_id)
    if not job: 
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed": 
        raise HTTPException(status_code=400, detail="Video not ready")
    return FileResponse(job["video_path"], media_type="video/mp4")

async def create_video(job_id: str, topic: str, duration: int):
    try:
        print(f"[{job_id}] Starting...")
        
        # Script (20%)
        update_job(job_id, {"progress": 20})
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Write a {duration}-second script about {topic}. {duration*2} words max."}]
        )
        script = response.choices[0].message.content
        
        # Audio (40%)
        update_job(job_id, {"progress": 40})
        tts = gTTS(text=script, lang='en', slow=False)
        audio_path = f"/tmp/audio_{job_id}.mp3"
        tts.save(audio_path)
        audio = AudioFileClip(audio_path)
        audio_duration = audio.duration
        
        caption_chunks = split_text_for_captions(script, max_words=5)
        chunk_duration = audio_duration / len(caption_chunks) if caption_chunks else 2
        
        # Video - SIMPLIFIED (Just use colored background to avoid crashes)
        update_job(job_id, {"progress": 60})
        print(f"[{job_id}] Creating video...")
        
        # Create simple colored background (reliable, no crashes)
        final_video = ColorClip(size=(720, 1280), color=(139, 92, 246), duration=audio_duration).with_fps(15)
        
        # Add captions
        update_job(job_id, {"progress": 75})
        if caption_chunks:
            caption_overlays = []
            for i, text in enumerate(caption_chunks):
                caption_img = create_caption_frame(text, width=680, height=150)
                if caption_img is not None:
                    img_clip = ImageClip(caption_img, transparent=True)
                    img_clip = img_clip.with_position(('center', 1050)).with_start(i * chunk_duration).with_duration(chunk_duration)
                    caption_overlays.append(img_clip)
            
            if caption_overlays:
                final_video = CompositeVideoClip([final_video] + caption_overlays, size=(720, 1280))
        
        # Audio only (no music to avoid complexity)
        update_job(job_id, {"progress": 85})
        final_video = final_video.with_audio(audio)
        
        # Render
        update_job(job_id, {"progress": 90})
        print(f"[{job_id}] Rendering...")
        
        output_path = f"/tmp/video_{job_id}.mp4"
        final_video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            preset="ultrafast",
            bitrate="500k",
            threads=1,
            logger=None
        )
        
        # Cleanup
        final_video.close()
        audio.close()
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        update_job(job_id, {"status": "completed", "progress": 100, "video_path": output_path})
        print(f"[{job_id}] SUCCESS!")
        
    except Exception as e:
        print(f"[{job_id}] FAILED: {e}")
        update_job(job_id, {"status": "failed", "error": str(e)})

# Startup cleanup
import shutil
if JOBS_DIR.exists():
    shutil.rmtree(JOBS_DIR)
    JOBS_DIR.mkdir(exist_ok=True)
