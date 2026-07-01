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
                        
                        # ✅ MoviePy 2.x: Use with_effects instead of crossfadein
                        if i > 0:
                            from moviepy import effects
                            clip = clip.with_effects([effects.CrossFadeIn(0.5)])
                        
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
                    from moviepy import ImageClip
                    img_clip = ImageClip(caption_img, transparent=True)
                    # ✅ MoviePy 2.x: Use with_position, with_start, with_duration
                    img_clip = img_clip.with_position(('center', 1050)).with_start(i * chunk_duration).with_duration(chunk_duration)
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
