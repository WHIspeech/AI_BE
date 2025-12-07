# ===========================================================
# WHISPEECH — Silent Lip to Intent → Sentence → TTS Pipeline
# 최종 안정판 (멀티프레임 FaceMesh + bytes input + Gradio 3.50.2 호환)
# ===========================================================

import os
import uuid
import cv2
import numpy as np
import torch
import gradio as gr
import mediapipe as mp
from gtts import gTTS
from dotenv import load_dotenv
import shutil

# -----------------------------------------------------------
# Gemini
# -----------------------------------------------------------
import google.generativeai as genai

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# -----------------------------------------------------------
# Intent Model
# -----------------------------------------------------------
from tiny_lip_intent_model import (
    TinyLipIntentNet,
    predict_intents
)
from intent_keyword_config import CANONICAL_KEYWORDS, TRIGGER_MAP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_INTENTS = len(CANONICAL_KEYWORDS)   # 반드시 학습 개수와 동일해야 함
MODEL_PATH = "tiny_lip_intent_best.pth"

intent_model = TinyLipIntentNet(num_intents=NUM_INTENTS, in_channels=3)
intent_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
intent_model.to(DEVICE)
intent_model.eval()

# -----------------------------------------------------------
# MediaPipe 설정
# -----------------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh

LIP_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308
]


# -----------------------------------------------------------
# 1) 가장 얼굴이 잘 잡히는 프레임 찾기 (멀티프레임)
# -----------------------------------------------------------
def find_best_mouth_frame(cap, total_frames):
    check_positions = [0.05, 0.25, 0.5, 0.75, 0.95]
    check_frames = [int(total_frames * p) for p in check_positions]

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        for idx in check_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                return frame

    return None


# -----------------------------------------------------------
# 2) 입 영역 박스 계산
# -----------------------------------------------------------
def get_mouth_box(frame, face_mesh, pad_x=1.2, pad_y=1.8):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0].landmark

    xs, ys = [], []
    for idx in LIP_LANDMARKS:
        xs.append(lm[idx].x * w)
        ys.append(lm[idx].y * h)

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    box_w = max_x - min_x
    box_h = max_y - min_y

    min_x = int(max(0, min_x - box_w * pad_x))
    max_x = int(min(w, max_x + box_w * pad_x))
    min_y = int(max(0, min_y - box_h * pad_y))
    max_y = int(min(h, max_y + box_h * pad_y))

    return min_x, min_y, max_x, max_y


# -----------------------------------------------------------
# 3) 영상 전체 crop
# -----------------------------------------------------------
def preprocess_video(in_path, out_path):

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        print("[SKIP] Cannot open video:", in_path)
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    best_frame = find_best_mouth_frame(cap, frame_count)
    if best_frame is None:
        print("[FAIL] No face detected in any frame")
        return None

    # detect mouth box
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        box = get_mouth_box(best_frame, face_mesh)
        if box is None:
            print("[FAIL] Mouth box not detected")
            return None

        x1, y1, x2, y2 = box

    cap.release()

    # crop whole video
    cap = cv2.VideoCapture(in_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(out_path, fourcc, fps, (x2 - x1, y2 - y1))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        crop = frame[y1:y2, x1:x2]
        out_writer.write(crop)
        frames.append(crop)

    cap.release()
    out_writer.release()

    return frames


# -----------------------------------------------------------
# 4) frames → npy
# -----------------------------------------------------------
def frames_to_npy(frames):
    os.makedirs("tmp_npy", exist_ok=True)
    arr = np.array(frames)
    npy_path = f"tmp_npy/{uuid.uuid4().hex}.npy"
    np.save(npy_path, arr)
    return npy_path


# -----------------------------------------------------------
# 5) Intent → 문장 생성
# -----------------------------------------------------------
def generate_sentence_from_intents(intent_list):
    if not intent_list:
        return "입모양에서 의도가 검출되지 않았습니다."

    tags = [x["intent"] for x in intent_list]

    prompt = f"""
입모양 기반 의도 태그를 자연스러운 한 문장 존댓말로 바꿔주세요.
태그: {tags}

규칙:
1) 새로운 정보 추가 금지
2) 태그 의미 안에서만 생성
3) 한 문장
"""

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        res = model.generate_content(prompt)
        return res.text.strip()
    except Exception as e:
        return f"문장 생성 실패: {e}"


# -----------------------------------------------------------
# 6) Sentence → TTS
# -----------------------------------------------------------
def generate_tts(text):
    os.makedirs("tts_outputs", exist_ok=True)
    out = f"tts_outputs/{uuid.uuid4().hex}.mp3"
    gTTS(text, lang="ko").save(out)
    return out


# -----------------------------------------------------------
# 7) 전체 파이프라인
# -----------------------------------------------------------
def full_pipeline(video_path):
    print("=== STEP 1: received path ===", video_path)

    safe_path = f"uploaded_videos/{uuid.uuid4().hex}.mp4"
    shutil.copy(video_path, safe_path)
    print("=== STEP 2: copied to ===", safe_path)

    cropped_path = f"tmp_video/{uuid.uuid4().hex}.mp4"
    print("=== STEP 3: start preprocess ===")

    frames = preprocess_video(safe_path, cropped_path)

    if frames is None:
        print("=== ERROR: preprocess failed ===")
        return {"error": "입 인식 실패"}, "", None

    print("=== STEP 4: frames extracted ===", len(frames))

    npy_path = frames_to_npy(frames)
    print("=== STEP 5: npy saved ===", npy_path)

    intents = predict_intents(
        model=intent_model,
        npy_path=npy_path,
        canonical_keywords=CANONICAL_KEYWORDS,
        device=DEVICE,
        top_k=3,
        threshold=0.3
    )
    print("=== STEP 6: intents ===", intents)

    sentence = generate_sentence_from_intents(intents)
    print("=== STEP 7: sentence ===", sentence)

    audio = generate_tts(sentence)
    print("=== STEP 8: tts saved ===", audio)

    return intents, sentence, audio

# -----------------------------------------------------------
# 8) Gradio (3.50.2 버전 기준)
# -----------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("## WHISPEECH — 묵음 발화 영상 복원 AI")

    video_in = gr.Video(label="입모양 영상 업로드(.mp4)")

    btn = gr.Button("복원하기")

    out_intents = gr.JSON(label="예측된 의도")
    out_sentence = gr.Textbox(label="복원된 문장")
    out_audio = gr.Audio(label="생성된 음성", type="filepath")

    btn.click(
        fn=full_pipeline,
        inputs=video_in,
        outputs=[out_intents, out_sentence, out_audio]
    )

if __name__ == "__main__":
    demo.launch()
