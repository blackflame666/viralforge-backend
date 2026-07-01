from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
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
    return {"status": "ViralForge API is running!"}

@app.post("/generate-video")
async def generate_video(topic: str):
    try:
        start_time = time.time()
        
        # 1. Generate script (keep it short - 10 seconds max)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Write ONE short engaging sentence about {topic} for a 10-second video. Maximum 20 words."}]
        )
        script = response.choices[0].message.content
        
        # 2. Generate voiceover
        tts = gTTS(text=script, lang='en', slow=False)
        audio_path = "audio.mp3"
        tts.save(audio_path)
        audio = AudioFileClip(audio_path)
        
        # 3. Get JUST ONE stock video from Pexels (faster)
        headers = {"Authorization": PEXELS_API_KEY}
        search_url = f"https://api.pexels.com/videos/search?query={topic}&per_page=1&orientation=portrait"
        pexels_res = requests.get(search_url, headers=headers, timeout=10).json()
        
        if not pexels_res['videos']:
            raise Exception("No videos found on Pexels")
        
        video = pexels_res['videos'][0]
        video_url = video['video_files'][0]['link']
        
        # Download video
        vid_res = requests.get(video_url, timeout=15)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
            tmp_vid.write(vid_res.content)
            clip = VideoFileClip(tmp_vid.name)
            
            # Simple: just loop the video to match audio length
            if clip.duration < audio.duration:
                clip = clip.loop(duration=audio.duration)
            else:
                clip = clip.subclipped(0, audio.duration)
            
            # Add audio
            final_clip = clip.with_audio(audio)
            
            # Write video (LOW quality for speed)
            output_path = "final_video.mp4"
            final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=15)
        
        elapsed = time.time() - start_time
        
        # Return file
        return FileResponse(output_path, media_type="video/mp4", filename=f"viralforge_{topic.replace(' ', '_')}.mp4")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
