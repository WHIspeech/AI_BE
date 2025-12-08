# ===========================================================
# WHISPEECH — FastAPI 웹 애플리케이션
# Silent Lip to Intent → Sentence → TTS Pipeline
# ===========================================================

import os
import uuid
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 기존 파이프라인 함수들 import
from ai_setence_tts_app import (
    full_pipeline,
    preprocess_video,
    frames_to_npy,
    generate_sentences_from_intents,
    generate_tts,
    intent_model,
    DEVICE,
    CANONICAL_KEYWORDS
)
from tiny_lip_intent_model import video_to_npy

# 필요한 폴더 생성
os.makedirs("uploaded_videos", exist_ok=True)
os.makedirs("tmp_video", exist_ok=True)
os.makedirs("tmp_npy", exist_ok=True)
os.makedirs("tts_outputs", exist_ok=True)
os.makedirs("static", exist_ok=True)

app = FastAPI(title="WHISPEECH - 묵음 발화 영상 복원 AI")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/tts_outputs", StaticFiles(directory="tts_outputs"), name="tts_outputs")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """메인 페이지"""
    html_path = Path("static/index.html")
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>WHISPEECH - 묵음 발화 영상 복원 AI</title>
        <meta charset="utf-8">
    </head>
    <body>
        <h1>WHISPEECH</h1>
        <p>HTML 파일이 static/index.html에 없습니다.</p>
    </body>
    </html>
    """


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    영상 파일 업로드 및 처리
    """
    # 파일 확장자 확인
    if not file.filename.endswith(('.mp4', '.MP4')):
        raise HTTPException(status_code=400, detail="MP4 파일만 업로드 가능합니다.")
    
    # 임시 파일로 저장
    file_id = uuid.uuid4().hex
    temp_path = f"uploaded_videos/{file_id}.mp4"
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 파이프라인 실행
        try:
            intents, sentences, audio_path = full_pipeline(temp_path)
            
            if isinstance(intents, dict) and "error" in intents:
                raise HTTPException(status_code=400, detail=intents["error"])
            
            # 오디오 파일 경로에서 파일명만 추출
            audio_filename = os.path.basename(audio_path) if audio_path else None
            
            return {
                "success": True,
                "intents": intents,
                "sentences": sentences,  # 문장 리스트 반환
                "audio_url": f"/tts_outputs/{audio_filename}" if audio_filename else None,
                "file_id": file_id
            }
        except Exception as e:
            # 임시 파일 정리
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"처리 중 오류 발생: {str(e)}")


@app.post("/api/generate_tts")
async def generate_tts_for_sentence(request: dict):
    """
    선택한 문장을 TTS로 변환
    """
    sentence = request.get("sentence")
    if not sentence:
        raise HTTPException(status_code=400, detail="문장이 제공되지 않았습니다.")
    
    try:
        audio_path = generate_tts(sentence)
        audio_filename = os.path.basename(audio_path) if audio_path else None
        
        return {
            "success": True,
            "audio_url": f"/tts_outputs/{audio_filename}" if audio_filename else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS 생성 실패: {str(e)}")


@app.get("/api/health")
async def health_check():
    """헬스 체크"""
    return {"status": "ok", "message": "WHISPEECH API is running"}


@app.get("/tts/{filename}")
async def get_audio(filename: str):
    """TTS 오디오 파일 다운로드"""
    file_path = f"tts_outputs/{filename}"
    if os.path.exists(file_path):
        return FileResponse(
            file_path,
            media_type="audio/mpeg",
            filename=filename
        )
    raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

