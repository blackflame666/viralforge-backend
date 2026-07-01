from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import requests
import tempfile
from openai import OpenAI
from gtts import gTTS
from moviepy import ColorClip, AudioFileClip, VideoFileClip, CompositeVideoClip, CompositeAudioClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
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

def create_caption_frame(text, width=720, height=200):
    """Create a caption image using PIL (no ImageMagick needed!)"""
    try:
        # Create black background with transparency
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Try to use a bold font, fallback to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", 40)
            except:
                font = ImageFont.load_default()
        
        # Get text bounding box for centering
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Calculate position (center horizontally)
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        # Draw text with black outline (stroke)
        stroke_width = 3
        draw.text((x-stroke_width, y), text, font=font, fill='black')
        draw.text((x+stroke_width, y), text, font=font, fill='black')
        draw.text((x, y-stroke_width), text, font=font, fill='black')
        draw.text((x, y+stroke_width), text, font=font, fill='black')
        
        # Draw white text on top
        draw.text((x, y), text, font=font, fill='white')
        
        # Convert to numpy array
        return np.array(img)
    except Exception as e:
        print(f"Caption creation error: {e}")
        return None

@app.get("/")
def root():
    return {"status": "ViralForge API v7 - Pillow Captions!"}

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

def split_text_for_captions(text, max_words=5):
    """Split script into chunks for captions"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = ' '.join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

async def download_background_music(duration, job_id):
    """Download background music from Pixabay"""
    try:
        if not PIXABAY_API_KEY:
            print(f"[{job_id}] ⚠️ No Pixabay API key found!")
            return None
        
        search_terms = ["upbeat", "motivational", "energetic", "happy"]
        import random
        term = random.choice(search_terms)
        
        url = f"https://pixabay.com/api/music/?key={PIXABAY_API_KEY}&q={term}&per_page=3"
        print(f"[{job_id}] Searching Pixabay for: {term}")
        res = requests.get(url, timeout=10)
        data = res.json()
        
        if data.get('hits') and len(data['hits']) > 0:
            track = data['hits'][0]  # Use first result
            music_url = track['preview']
            print(f"[{job_id}] 🎵 Found music: {track['title']}")
            
            music_data = requests.get(music_url, timeout=15).content
            music_path = f"/tmp/music_{job_id}.mp3"
            
            with open(music_path, 'wb') as f:
                f.write(music_data)
            
            print(f"[{job_id}] ✅ Music downloaded!")
            return music_path
        else:
            print(f"[{job_id}] ❌ No music found on Pixabay")
            return None
            
    except Exception as e:
        print(f"[{job_id}] ❌ Music error: {e}")
        return None

async def create_video(job_id: str, topic: str, duration: int):
    music_path = None
    try:
        print(f"[{job_id}] 🎬 Starting Pro generation...")
        
        # 20% - Script
        update_job(job_id, {"progress": 20})
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Write a {duration}-second viral script about {topic}. {duration*2} words max."}]
        )
        script = response.choices[0].message.content
        print(f"[{job_id}] 📝 Script: {script[:50]}...")
        
        # 40% - Audio
        update_job(job_id, {"progress": 40})
        tts = gTTS(text=script, lang='en', slow=False)
        audio_path = f"/tmp/audio_{job_id}.mp3"
        tts.save(audio_path)
        audio = AudioFileClip(audio_path)
        audio_duration = audio.duration
        print(f"[{job_id}] 🎤 Audio duration: {audio_duration:.2f}s")
        
        caption_chunks = split_text_for_captions(script, max_words=5)
        chunk_duration = audio_duration / len(caption_chunks) if caption_chunks else 2
        
        # 60% - Fetch Stock Footage
        update_job(job_id, {"progress": 60})
        print(f"[{job_id}] 🎥 Fetching stock footage...")
        
        video_clips = []
        if PEXELS_API_KEY:
            try:
                headers = {"Authorization": PEXELS_API_KEY}
                url = f"https://api.pexels.com/videos/search?query={topic}&per_page=2&orientation=portrait"
                res = requests.get(url, headers=headers, timeout=10).json()
                
                if res.get('videos') and len(res['videos']) > 0:
                    for i, video in enumerate(res['videos'][:2]):
                        video_url = video['video_files'][0]['link']
                        vid_data = requests.get(video_url, timeout=15).content
                        
                        temp_path = f"/tmp/stock_{job_id}_{i}.mp4"
                        with open(temp_path, 'wb') as f:
                            f.write(vid_data)
                        
                        clip = VideoFileClip(temp_path)
                        clip = clip.resized((720, 1280))
                        
                        clip_dur = audio_duration / 2
                        if clip.duration < clip_dur:
                            clip = clip.loop(duration=clip_dur)
                        else:
                            clip = clip.subclipped(0, clip_dur)
                        
                        if i > 0:
                            clip = clip.crossfadein(0.5)
                        
                        video_clips.append(clip)
                        print(f"[{job_id}] ✅ Clip {i+1} ready")
                else:
                    print(f"[{job_id}] ⚠️ No videos found")
                    
            except Exception as e:
                print(f"[{job_id}] ❌ Pexels error: {e}")
        else:
            print(f"[{job_id}] ⚠️ No Pexels API Key")
        
        # 70% - Download Background Music
        update_job(job_id, {"progress": 70})
        music_path = await download_background_music(audio_duration, job_id)
        
        # 75% - Create Video with Captions using PIL
        update_job(job_id, {"progress": 75})
        print(f"[{job_id}] 📝 Creating captions with PIL...")
        
        # Combine base video
        if video_clips:
            final_video = CompositeVideoClip(video_clips, size=(720, 1280))
        else:
            final_video = ColorClip(size=(720, 1280), color=(139, 92, 246), duration=audio_duration).with_fps(15)
        
        # Add captions as overlay
        if caption_chunks:
            print(f"[{job_id}] Adding {len(caption_chunks)} caption overlays...")
            caption_overlays = []
            
            for i, text in enumerate(caption_chunks):
                caption_img = create_caption_frame(text, width=680, height=150)
                if caption_img is not None:
                    # Create ImageClip from numpy array
                    from moviepy import ImageClip
                    img_clip = ImageClip(caption_img, transparent=True)
                    # Position at bottom center
                    img_clip = img_clip.set_position(('center', 1050)).set_start(i * chunk_duration).set_duration(chunk_duration)
                    caption_overlays.append(img_clip)
            
            if caption_overlays:
                final_video = CompositeVideoClip([final_video] + caption_overlays, size=(720, 1280))
                print(f"[{job_id}] ✅ Captions added!")
            else:
                print(f"[{job_id}] ⚠️ No captions created")
        
        final_video = final_video.with_fps(15)
        
        # 85% - Mix Audio
        update_job(job_id, {"progress": 85})
        print(f"[{job_id}] 🎵 Mixing audio...")
        
        if music_path and os.path.exists(music_path):
            try:
                bg_music = AudioFileClip(music_path)
                
                if bg_music.duration < audio_duration:
                    bg_music = bg_music.loop(duration=audio_duration)
                else:
                    bg_music = bg_music.subclipped(0, audio_duration)
                
                bg_music = bg_music.volumex(0.25)
                final_audio = CompositeAudioClip([audio, bg_music])
                bg_music.close()
                print(f"[{job_id}] ✅ Audio mixed with music!")
                
            except Exception as e:
                print(f"[{job_id}] ❌ Music mixing error: {e}")
                final_audio = audio
        else:
            print(f"[{job_id}] ⚠️ Using voiceover only (no music)")
            final_audio = audio
        
        final_video = final_video.with_audio(final_audio)
        
        # 90% - Render
        update_job(job_id, {"progress": 90})
        print(f"[{job_id}] 🎬 Rendering...")
        
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
        
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if music_path and os.path.exists(music_path):
            os.remove(music_path)
        
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
        
        print(f"[{job_id}] ✅ SUCCESS with music and captions!")
        
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
