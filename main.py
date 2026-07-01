from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import requests
import tempfile
from openai import OpenAI
from gtts import gTTS
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips
import imageio_ffmpeg
import time

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

@app.get("/")
def root():
    return {"status": "ViralForge API is running!", "plan": "Pro"}

@app.post("/generate-video")
async def generate_video(topic: str, duration: int = 30):
    try:
        start_time = time.time()
        
        # 1. Generate LONGER script (30-60 seconds)
        word_count = duration * 2  # ~2 words per second
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Write an engaging {duration}-second TikTok script about {topic}. Use exactly {word_count} words. Make it exciting and viral-worthy. Just the spoken text, no intro/outro."}]
        )
        script = response.choices[0].message.content
        
        # 2. Generate voiceover
        tts = gTTS(text=script, lang='en', slow=False)
        audio_path = "audio.mp3"
        tts.save(audio_path)
        audio = AudioFileClip(audio_path)
        
        # 3. Get MULTIPLE stock videos from Pexels
        headers = {"Authorization": PEXELS_API_KEY}
        search_url = f"https://api.pexels.com/videos/search?query={topic}&per_page=5&orientation=portrait"
        pexels_res = requests.get(search_url, headers=headers, timeout=15).json()
        
        video_clips = []
        clip_duration = audio.duration / len(pexels_res['videos'][:3])
        
        for video in pexels_res['videos'][:3]:
            video_url = video['video_files'][0]['link']
            vid_res = requests.get(video_url, timeout=20)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
                tmp_vid.write(vid_res.content)
                clip = VideoFileClip(tmp_vid.name)
                
                # Resize to vertical 1080x1920 for TikTok
                clip = clip.resized((1080, 1920))
                
                # Loop or trim to match duration
                if clip.duration < clip_duration:
                    clip = clip.loop(duration=clip_duration)
                else:
                    clip = clip.subclipped(0, clip_duration)
                
                video_clips.append(clip)
        
        # 4. Combine all clips
        final_video = concatenate_videoclips(video_clips, method="compose")
        final_video = final_video.with_audio(audio)
        
        # 5. Export HIGH QUALITY video
        output_path = "final_video.mp4"
        final_video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=24,
            preset="medium",
            bitrate="5000k"
        )
        
        elapsed = time.time() - start_time
        
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=f"viralforge_{topic.replace(' ', '_')}_{duration}s.mp4"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
