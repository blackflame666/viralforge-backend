from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
from openai import OpenAI
from gtts import gTTS

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
        # 1. Generate script with OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Write a short, engaging 30-second TikTok script about {topic}. Just give me the spoken text, no intro or outro."}]
        )
        script = response.choices[0].message.content
        
        # 2. Generate voiceover with gTTS (Google Text-to-Speech)
        tts = gTTS(text=script, lang='en', slow=False)
        audio_path = "audio.mp3"
        tts.save(audio_path)
        
        return {
            "status": "success",
            "script": script,
            "message": "Audio generated successfully!",
            "audio_length": "Check logs for duration"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
