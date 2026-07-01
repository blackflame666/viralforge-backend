from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import requests
import tempfile
from openai import OpenAI
from gtts import gTTS
from moviepy import ColorClip, AudioFileClip, VideoFileClip, CompositeVideoClip, CompositeAudioClip, TextClip
import imageio_ffmpeg
import asyncio
import time
import uuid
import json
from pathlib import Path
import re

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
    return {"status": "ViralForge API v5 - Pro Features!"}

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

def split_text_for_captions(text, max_words=6):
    """Split script into chunks for captions"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = ' '.join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

async def create_video(job_id: str, topic: str, duration: int):
    stock_video_path = None
    bg_music_path = None
    try:
        print(f"[{job_id}] Starting Pro generation...")
        
        # 20% - Script
        update_job(job_id, {"progress": 20})
        print(f"[{job_id}] Generating script...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Write a {duration}-second viral script about {topic}. {duration*2} words max. Make it engaging with short punchy sentences."}]
        )
        script = response.choices[0].message.content
        print(f"[{job_id}] Script: {script[:50]}...")
        
        # 40% - Audio
        update_job(job_id, {"progress": 40})
        print(f"[{job_id}] Generating voiceover...")
        tts = gTTS(text=script, lang='en', slow=False)
        audio_path = f"/tmp/audio_{job_id}.mp3"
        tts.save(audio_path)
        audio = AudioFileClip(audio_path)
        audio_duration = audio.duration
        
        # Split script for captions
        caption_chunks = split_text_for_captions(script, max_words=5)
        chunk_duration = audio_duration / len(caption_chunks) if caption_chunks else 2
        
        # 60% - Fetch Stock Footage
        update_job(job_id, {"progress": 60})
        print(f"[{job_id}] Fetching stock footage...")
        
        if PEXELS_API_KEY:
            try:
                headers = {"Authorization": PEXELS_API_KEY}
                url = f"https://api.pexels.com/videos/search?query={topic}&per_page=2&orientation=portrait"
                res = requests.get(url, headers=headers, timeout=10).json()
                
                video_clips = []
                
                if res.get('videos') and len(res['videos']) > 0:
                    for i, video in enumerate(res['videos'][:2]):
                        video_url = video['video_files'][0]['link']
                        print(f"[{job_id}] Downloading clip {i+1}...")
                        vid_data = requests.get(video_url, timeout=15).content
                        
                        temp_path = f"/tmp/stock_{job_id}_{i}.mp4"
                        with open(temp_path, 'wb') as f:
                            f.write(vid_data)
                        
                        clip = VideoFileClip(temp_path)
                        clip = clip.resized((720, 1280))
                        
                        # Add transition (crossfade)
                        clip_duration = audio_duration / 2
                        if clip.duration < clip_duration:
                            clip = clip.loop(duration=clip_duration)
                        else:
                            clip = clip.subclipped(0, clip_duration)
                        
                        # Add crossfade transition
                        if i > 0:
                            clip = clip.crossfadein(0.5)
                        
                        video_clips.append(clip)
                else:
                    print(f"[{job_id}] No videos found")
                    
            except Exception as e:
                print(f"[{job_id}] Pexels error: {e}")
        else:
            print(f"[{job_id}] No Pexels API Key")
            video_clips = []
        
        # 70% - Create Video with Captions
        update_job(job_id, {"progress": 70})
        print(f"[{job_id}] Adding captions and effects...")
        
        # Create captions
        caption_clips = []
        for i, text in enumerate(caption_chunks):
            try:
                # Create text clip
                txt_clip = TextClip(
                    text=text,
                    size=(680, 200),
                    font='Arial',
                    fontsize=50,
                    color='white',
                    stroke_color='black',
                    stroke_width=2,
                    method='caption'
                )
                
                # Position at bottom center
                txt_clip = txt_clip.set_position(('center', 1000)).set_start(i * chunk_duration).set_duration(chunk_duration)
                caption_clips.append(txt_clip)
            except Exception as e:
                print(f"[{job_id}] Caption error: {e}")
        
        # Combine video clips
        if video_clips:
            final_video = CompositeVideoClip(video_clips, size=(720, 1280))
        else:
            # Fallback to purple gradient
            final_video = ColorClip(size=(720, 1280), color=(139, 92, 246), duration=audio_duration).with_fps(15)
        
        # Add captions on top
        if caption_clips:
            final_video = CompositeVideoClip([final_video] + caption_clips, size=(720, 1280))
        
        final_video = final_video.with_fps(15)
        
        # 80% - Background Music (Optional)
        update_job(job_id, {"progress": 80})
        print(f"[{job_id}] Mixing audio...")
        
        # Try to get a free background music (royalty-free)
        try:
            # Using a simple royalty-free music URL or create a simple tone
            # For now, we'll skip external music to avoid errors
            # You can add your own music file path here
            final_audio = audio
        except Exception as e:
            print(f"[{job_id}] Music error: {e}")
            final_audio = audio
        
        final_video = final_video.with_audio(final_audio)
        
        # 90% - Render
        update_job(job_id, {"progress": 90})
        print(f"[{job_id}] Rendering final video...")
        
        output_path = f"/tmp/video_{job_id}.mp4"
        final_video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            preset="ultrafast",
            bitrate="800k",
            threads=1,
            logger=None
        )
        
        # Cleanup
        final_video.close()
        audio.close()
        if stock_video_path and os.path.exists(stock_video_path):
            os.remove(stock_video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)
        # Clean up temp video files
        for i in range(3):
            temp_file = f"/tmp/stock_{job_id}_{i}.mp4"
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
        # 100% - Done
        update_job(job_id, {
            "status": "completed", 
            "progress": 100, 
            "video_path": output_path
        })
        
        print(f"[{job_id}] ✅ SUCCESS with captions!")
        
    except Exception as e:
        error_msg = str(e)
        print(f"[{job_id}] ❌ FAILED: {error_msg}")
        import traceback
        traceback.print_exc()
        update_job(job_id, {
            "status": "failed", 
            "error": error_msg
        })

# Cleanup
import shutil
if JOBS_DIR.exists():
    shutil.rmtree(JOBS_DIR)
    JOBS_DIR.mkdir(exist_ok=True)
