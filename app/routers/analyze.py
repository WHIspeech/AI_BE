from fastapi import APIRouter, UploadFile, File
import uuid
import os

from app.services.preprocessing import preprocess_video, frames_to_npy
from app.services.predictor import load_intent_model, predict_intents
from app.services.sentence import generate_sentence_from_intents
from app.services.tts import generate_tts

router = APIRouter()
intent_model, keywords, device = load_intent_model()


@router.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    # 1) 저장
    upload_dir = "app/static/uploads"
    os.makedirs(upload_dir, exist_ok=True)

    file_id = uuid.uuid4().hex
    input_path = f"{upload_dir}/{file_id}.mp4"

    with open(input_path, "wb") as f:
        f.write(await file.read())

    # 2) 전처리
    frames = preprocess_video(input_path)
    if frames is None:
        return {"error": "입 영역 인식 실패"}

    # 3) npy로 변환
    npy_path = frames_to_npy(frames)

    # 4) intent 예측
    intents = predict_intents(intent_model, npy_path, keywords, device)

    # 5) 문장 생성
    sentence = generate_sentence_from_intents(intents)

    # 6) tts 생성
    audio_path = generate_tts(sentence)

    return {
        "intents": intents,
        "sentence": sentence,
        "audio": audio_path.replace("app/", "")  # static 경로만 반환
    }
