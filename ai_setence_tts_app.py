# ===========================================================
# WHISPEECH — Silent Lip to Intent → Sentence → TTS Pipeline
# 최종 안정판 (멀티프레임 FaceMesh + bytes input + Gradio 3.50.2 대응)
# ===========================================================
print("=== SCRIPT STARTED ===")
import os
import uuid
import cv2
import numpy as np
import torch
import mediapipe as mp
from gtts import gTTS
from dotenv import load_dotenv
import shutil
import base64  # ✅ 추가

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
    predict_intents as predict_intents_from_arr
)
from intent_keyword_config import CANONICAL_KEYWORDS, TRIGGER_MAP

# npy_path를 받는 predict_intents 래퍼 함수
def predict_intents(model, npy_path, canonical_keywords, top_k=3, threshold=0.3, device="cpu"):
    """npy_path를 받아서 배열로 변환 후 predict_intents 호출"""
    arr = np.load(npy_path)
    
    # 데이터 타입 확인 및 변환 (uint8 → float32)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
    
    return predict_intents_from_arr(model, arr, canonical_keywords, top_k, threshold, device)

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

    # MediaPipe 초기화 (오류 발생 시 최소 옵션으로 재시도)
    try:
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    except Exception as e:
        print(f"[WARNING] FaceMesh 초기화 실패, 최소 옵션으로 재시도: {e}")
        try:
            face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.5
            )
        except Exception as e2:
            print(f"[ERROR] FaceMesh 초기화 완전 실패: {e2}")
            return None

    try:
        for idx in check_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                face_mesh.close()
                return frame
    finally:
        try:
            face_mesh.close()
        except:
            pass

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
    try:
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    except Exception as e:
        print(f"[WARNING] FaceMesh 초기화 실패, 최소 옵션으로 재시도: {e}")
        try:
            face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.5
            )
        except Exception as e2:
            print(f"[ERROR] FaceMesh 초기화 완전 실패: {e2}")
            return None

    try:
        box = get_mouth_box(best_frame, face_mesh)
    finally:
        try:
            face_mesh.close()
        except:
            pass
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
    
    # (T, H, W, C) → (C, T, H, W) 변환
    if len(arr.shape) == 4:  # (T, H, W, C)
        arr = arr.transpose(3, 0, 1, 2)  # (C, T, H, W)
    
    # uint8 → float32 변환 및 정규화 [0, 1]
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
    
    npy_path = f"tmp_npy/{uuid.uuid4().hex}.npy"
    np.save(npy_path, arr)
    return npy_path


# -----------------------------------------------------------
# 5) Intent → 문장 후보 생성 (3개)
# -----------------------------------------------------------
def generate_spoken_sentence_candidates(intent_list):
    """
    의도 태그들을 조합하여
    '실제로 영상 속 사람이 말했을 법한 자연스러운 문장 후보 3개'
    를 생성하는 함수.
    """

    if not intent_list:
        return ["입모양에서 의미 있는 문장을 복원할 수 없습니다."]

    # intent → 한글 변환
    intent_to_korean = {
        "TRAVEL": "여행",
        "SCHEDULE": "일정",
        "PLACE_NATURE": "자연 장소",
        "PLACE_CITY": "도시",
        "COMPANION": "동행",
        "FOOD_DRINK": "음식과 음료",
        "TRANSPORT": "교통",
        "COST": "비용",
        "TIME": "시간",
        "ACTIVITY": "활동",
        "EMOTION_POS": "긍정적 감정",
        "EMOTION_NEG": "부정적 감정",
        "WATER": "물",
        "FOOD_CARE": "음식 케어",
        "TOILET": "화장실",
        "PAIN": "통증",
        "HELP": "도움",
        "MOVE": "이동"
    }

    tags = [item["intent"] for item in intent_list]
    kor_tags = [intent_to_korean.get(t, t) for t in tags]

    tags_str = ", ".join(tags)
    kor_str = ", ".join(kor_tags)

    prompt = f"""
당신은 '입모양 기반 문장 복원 전문가'입니다.

아래 의도 태그들을 기반으로,
실제로 사람이 말했을 법한 자연스러운 한국어 문장 후보 3개를 생성하세요.

의도 태그: {tags_str}
의도 의미(한글): {kor_str}

규칙:
1) 태그 의미 범위 안에서만 문장 생성 (새 정보 추가 금지)
2) 매우 자연스럽고 현실적인 문장
3) 존댓말 한 문장
4) 불릿포인트·번호·인용금지
5) 세 문장을 줄바꿈으로만 구분
6) “~말씀하시는 것 같습니다” 같은 패턴 금지 → 실제 발화 형태

예시:
- TRAVEL + TIME → “언제 여행 갈 수 있을까요?”
- WATER → “혹시 물 좀 주실 수 있나요?”
- TOILET + HELP → “화장실이 어딘지 도와주실 수 있을까요?”

출력 예:
문장1
문장2
문장3
"""

    try:
        # 모델 우선순위
        models_to_try = [
            "models/gemini-2.5-flash",
            "models/gemini-flash-latest",
            "models/gemini-2.0-flash",
            "models/gemini-2.5-pro",
            "models/gemini-pro-latest",
            "gemini-2.0-flash",
            "gemini-2.0-pro",
            "gemini-1.5-flash",
            "gemini-1.5-pro"
        ]

        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                res = model.generate_content(prompt)
                if res and res.text:
                    text = res.text.strip()
                    # bullet/번호 제거
                    lines = [
                        l.lstrip("0123456789.-•)").strip()
                        for l in text.split("\n")
                        if len(l.strip()) > 3
                    ]
                    # 최대 3개만 반환
                    return lines[:3]
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    break
                continue

    except Exception:
        pass

    # fallback
    return [
        f"{kor_str}에 대해 말하시는 듯합니다.",
        f"{kor_str} 관련해 여쭤보시는 것 같습니다.",
        f"{kor_str}에 대한 궁금한 점이 있으신가요?"
    ]



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
def full_pipeline(video_input):
    print("\n========== FULL PIPELINE CALLED ==========")
    print("RAW INPUT TYPE:", type(video_input))
    print("RAW INPUT VALUE:", video_input)

    try:
        # 1) 업로드된 비디오를 안전한 파일로 저장
        os.makedirs("uploaded_videos", exist_ok=True)

        if isinstance(video_input, dict):
            # Gradio 3.50.2: {'name': ..., 'data': 'data:video/mp4;base64,....'}
            data = video_input.get("data")
            if not data:
                return "비디오 데이터가 비어 있습니다.", "", ""

            # "data:video/mp4;base64,xxxx" → "xxxx"만 떼기
            if "," in data:
                b64 = data.split(",", 1)[1]
            else:
                b64 = data

            video_bytes = base64.b64decode(b64)
            safe_path = f"uploaded_videos/{uuid.uuid4().hex}.mp4"
            with open(safe_path, "wb") as f:
                f.write(video_bytes)

        elif isinstance(video_input, str):
            # 일부 환경에서는 파일 경로(str)로 들어옴
            safe_path = f"uploaded_videos/{uuid.uuid4().hex}.mp4"
            shutil.copy(video_input, safe_path)
        else:
            return f"지원하지 않는 입력 타입입니다: {type(video_input)}", "", ""

        print("=== STEP 1: saved to ===", safe_path)

        # 2) crop
        os.makedirs("tmp_video", exist_ok=True)
        cropped_path = f"tmp_video/{uuid.uuid4().hex}.mp4"
        print("=== STEP 2: start preprocess ===")

        frames = preprocess_video(safe_path, cropped_path)

        if frames is None:
            print("=== ERROR: preprocess failed ===")
            return "입 인식 실패 (Mediapipe에서 얼굴/입을 못 찾았습니다.)", "", ""

        print("=== STEP 3: frames extracted ===", len(frames))

        # 3) npy
        npy_path = frames_to_npy(frames)
        print("=== STEP 4: npy saved ===", npy_path)

        # 4) Intent
        intents = predict_intents(
            model=intent_model,
            npy_path=npy_path,
            canonical_keywords=CANONICAL_KEYWORDS,
            device=DEVICE,
            top_k=3,
            threshold=0.3
        )
        print("=== STEP 5: intents ===", intents)

        # 5) Sentences (multiple sentences, one per intent)
        sentences = generate_spoken_sentence_candidates(intents)
        print("=== STEP 6: sentences ===", sentences)

        # 6) Audio (first sentence as default)
        default_audio = generate_tts(sentences[0] if sentences else "의도가 검출되지 않았습니다.")
        print("=== STEP 7: default tts saved ===", default_audio)

        return intents, sentences, default_audio

    except Exception as e:
        import traceback
        traceback.print_exc()
        # 에러 내용을 첫 번째 반환값에 보여주자
        return f"에러 발생: {e}", "", ""

# -----------------------------------------------------------
# 참고: 이 파일은 파이프라인 함수들을 제공합니다.
# FastAPI 웹 애플리케이션은 app.py를 사용하세요.
# -----------------------------------------------------------