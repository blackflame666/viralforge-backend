from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import asyncio
import edge_tts
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
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
            messages=[{"role": "user", "content": f"Write a short TikTok script about {topic}"}]
        )
        script = response.choices[0].message.content
        
        # 2. Generate voiceover
        communicate = edge_tts.Communicate(script, "en-US-GuyNeural")
        audio_path = "audio.mp3"
        await communicate.save(audio_path)
        
        return {
            "status": "success",
            "script": script,
            "message": "Audio generated! (Video generation coming soon)"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
