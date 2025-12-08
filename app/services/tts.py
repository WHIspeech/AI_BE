from gtts import gTTS
import uuid
import os

def generate_tts(text):
    path = f"app/static/tts/{uuid.uuid4().hex}.mp3"
    os.makedirs("app/static/tts", exist_ok=True)
    gTTS(text, lang="ko").save(path)
    return path
