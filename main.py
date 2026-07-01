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

# Tell MoviePy where the FFmpeg binary is
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
    return {"status": "ViralForge API is running!"}

@app.post("/generate-video")
async def generate_video(topic: str):
    try:
        # 1. Generate script
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Write a short, engaging 15-second TikTok script about {topic}. Just give me the spoken text, no intro or outro."}]
        )
        script = response.choices[0].message.content
        
        # 2. Generate voiceover
        tts = gTTS(text=script, lang='en', slow=False)
        audio_path = "audio.mp3"
        tts.save(audio_path)
        audio = AudioFileClip(audio_path)
        
        # 3. Get stock footage from Pexels
        headers = {"Authorization": PEXELS_API_KEY}
        search_url = f"https://api.pexels.com/videos/search?query={topic}&per_page=3&orientation=portrait"
        pexels_res = requests.get(search_url, headers=headers).json()
        
        video_clips = []
        clip_duration = audio.duration / 3 
        
        for video in pexels_res['videos'][:3]:
            video_url = video['video_files'][0]['link']
            vid_res = requests.get(video_url)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
                tmp_vid.write(vid_res.content)
                clip = VideoFileClip(tmp_vid.name)
                # Resize to vertical 1080x1920 for TikTok
                clip = clip.resized((1080, 1920))
                
                # Loop or trim clip to match required duration
                if clip.duration < clip_duration:
                    clip = clip.loop(duration=clip_duration)
                else:
                    clip = clip.subclipped(0, clip_duration)
                video_clips.append(clip)
        
        # 4. Combine video and audio
        final_video = concatenate_videoclips(video_clips, method="compose")
        final_video = final_video.with_audio(audio)
        
        output_path = "final_video.mp4"
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=24)
        
        return FileResponse(output_path, media_type="video/mp4", filename="viralforge_video.mp4")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
