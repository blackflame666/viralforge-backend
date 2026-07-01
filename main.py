from fastapi import FastAPI, HTTPException, Query
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
async def generate_video(
    topic: str = Query(..., description="Video topic"),
    duration: int = Query(default=15, ge=10, le=60, description="Duration in seconds (10-60)")
):
    try:
        start_time = time.time()
        
        # 1. Generate script
        word_count = min(duration * 2, 60)  # Cap at 60 words for speed
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Write an engaging {duration}-second script about {topic}. Use {word_count} words max. Be concise and viral."}]
        )
        script = response.choices[0].message.content
        
        # 2. Generate voiceover
        tts = gTTS(text=script, lang='en', slow=False)
        audio_path = "audio.mp3"
        tts.save(audio_path)
        audio = AudioFileClip(audio_path)
        
        # 3. Get stock footage (just 2 clips for speed)
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
        
        # 4. Combine and render (FASTER settings)
        final_video = concatenate_videoclips(video_clips, method="compose")
        final_video = final_video.with_audio(audio)
        
        output_path = "final_video.mp4"
        final_video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=24,
            preset="ultrafast",  # Much faster rendering
            bitrate="2000k"      # Lower bitrate = faster render
        )
        
        elapsed = time.time() - start_time
        print(f"Video generated in {elapsed:.2f} seconds")
        
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=f"viralforge_{topic.replace(' ', '_')}.mp4"
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
