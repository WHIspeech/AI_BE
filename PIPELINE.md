# WHISPEECH 파이프라인 상세 가이드

## 📋 목차
1. [파이프라인 개요](#파이프라인-개요)
2. [전체 흐름도](#전체-흐름도)
3. [단계별 상세 설명](#단계별-상세-설명)
4. [데이터 형식](#데이터-형식)
5. [에러 처리](#에러-처리)
6. [성능 최적화](#성능-최적화)

---

## 파이프라인 개요

WHISPEECH 파이프라인은 **묵음 상태의 입모양 영상**을 입력받아 **의도 예측 → 문장 생성 → 음성 변환**까지의 전체 프로세스를 수행합니다.

### 핵심 함수
```python
full_pipeline(video_input) → (intents, sentences, audio_path)
```

### 입력 형식
- **Gradio 3.50.2**: `{'name': '...', 'data': 'data:video/mp4;base64,...'}`
- **FastAPI**: 파일 경로 문자열 (`str`)
- **직접 호출**: 파일 경로 문자열

### 출력 형식
```python
(
    intents: List[Dict[str, Any]],      # [{"intent": "TRAVEL", "score": 0.95}, ...]
    sentences: List[str],               # ["여행을 계획하고 있습니다.", ...]
    audio_path: str                     # "tts_outputs/abc123.mp3"
)
```

---

## 전체 흐름도

```
┌─────────────────────────────────────────────────────────────────┐
│                    WHISPEECH 파이프라인                          │
└─────────────────────────────────────────────────────────────────┘

[1단계] 비디오 입력
    │
    ├─ Base64 디코딩 (Gradio) 또는 파일 복사 (FastAPI)
    │
    └─→ uploaded_videos/{uuid}.mp4

[2단계] 전처리 (Preprocessing)
    │
    ├─ find_best_mouth_frame()      # 최적 프레임 찾기
    ├─ get_mouth_box()               # 입 영역 박스 계산
    ├─ preprocess_video()            # 전체 영상 크롭
    │
    └─→ frames[] (List[np.ndarray])  # 크롭된 프레임 리스트

[3단계] NumPy 변환
    │
    ├─ frames_to_npy()
    ├─ (T, H, W, C) → (C, T, H, W) 변환
    ├─ uint8 → float32, [0, 255] → [0, 1] 정규화
    │
    └─→ tmp_npy/{uuid}.npy           # (C=3, T=40, H=112, W=112)

[4단계] 의도 예측 (Intent Prediction)
    │
    ├─ predict_intents()
    ├─ TinyLipIntentNet 모델 추론
    ├─ Sigmoid → 확률 점수
    ├─ Threshold(0.3) 또는 Top-K(3) 선택
    │
    └─→ intents[]                    # [{"intent": "TRAVEL", "score": 0.95}, ...]

[5단계] 문장 생성 (Sentence Generation)
    │
    ├─ generate_sentences_from_intents()
    ├─ 각 의도마다 Gemini API 호출
    ├─ Fallback 메커니즘 (API 실패 시)
    │
    └─→ sentences[]                  # ["여행을 계획하고 있습니다.", ...]

[6단계] TTS 생성 (Text-to-Speech)
    │
    ├─ generate_tts()
    ├─ gTTS (Google Text-to-Speech)
    │
    └─→ tts_outputs/{uuid}.mp3       # 음성 파일

[최종 출력]
    │
    └─→ (intents, sentences, audio_path)
```

---

## 단계별 상세 설명

### 1단계: 비디오 입력 및 저장

#### 목적
다양한 입력 형식을 안전하게 처리하여 로컬 파일로 저장

#### 처리 과정

**1.1 입력 타입 확인**
```python
if isinstance(video_input, dict):
    # Gradio 3.50.2 형식
    data = video_input.get("data")
    # "data:video/mp4;base64,xxxx" → "xxxx" 추출
    b64 = data.split(",", 1)[1]
    video_bytes = base64.b64decode(b64)
    
elif isinstance(video_input, str):
    # 파일 경로 문자열
    shutil.copy(video_input, safe_path)
```

**1.2 안전한 파일 저장**
- 디렉토리: `uploaded_videos/`
- 파일명: UUID 기반 고유 이름 (`{uuid}.mp4`)
- 목적: 동시 요청 처리 및 파일 충돌 방지

#### 출력
- **파일 경로**: `uploaded_videos/{uuid}.mp4`
- **형식**: MP4 비디오 파일

#### 에러 처리
- 비어있는 데이터 → `"비디오 데이터가 비어 있습니다."`
- 지원하지 않는 타입 → `"지원하지 않는 입력 타입입니다: {type}"`

---

### 2단계: 전처리 (Preprocessing)

#### 목적
영상에서 입 영역만 추출하여 모델 입력에 적합한 형태로 변환

#### 처리 과정

**2.1 최적 프레임 찾기** (`find_best_mouth_frame()`)

```python
check_positions = [0.05, 0.25, 0.5, 0.75, 0.95]
check_frames = [int(total_frames * p) for p in check_positions]
```

- **전략**: 영상의 5개 위치(5%, 25%, 50%, 75%, 95%)에서 얼굴 감지 시도
- **이유**: 초반/후반 프레임에서 얼굴이 안 보일 수 있음
- **MediaPipe 설정**:
  ```python
  FaceMesh(
      static_image_mode=True,
      max_num_faces=1,
      refine_landmarks=False,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5
  )
  ```
- **Fallback**: 초기화 실패 시 최소 옵션으로 재시도

**2.2 입 영역 박스 계산** (`get_mouth_box()`)

```python
LIP_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308
]
```

- **랜드마크**: MediaPipe FaceMesh의 입 주변 23개 포인트
- **박스 계산**:
  1. 랜드마크 좌표 수집
  2. 최소/최대 x, y 좌표 계산
  3. 패딩 적용:
     - `pad_x = 1.2` (좌우 20% 확장)
     - `pad_y = 1.8` (상하 80% 확장)
  4. 이미지 경계 내로 제한

**2.3 전체 영상 크롭** (`preprocess_video()`)

```python
# 모든 프레임에 동일한 박스 적용
for frame in video:
    crop = frame[y1:y2, x1:x2]
    frames.append(crop)
```

- **일관성**: 한 프레임에서 계산한 박스를 모든 프레임에 적용
- **출력**: 크롭된 프레임 리스트 + 크롭 영상 파일 저장

#### 출력
- **프레임 리스트**: `List[np.ndarray]` - 각 프레임은 (H, W, 3) 형태
- **크롭 영상**: `tmp_video/{uuid}.mp4` (선택적)

#### 에러 처리
- 영상 열기 실패 → `None` 반환
- 얼굴 감지 실패 → `None` 반환
- 입 박스 계산 실패 → `None` 반환

---

### 3단계: NumPy 변환

#### 목적
프레임 리스트를 모델 입력 형식으로 변환

#### 처리 과정

**3.1 배열 변환**
```python
arr = np.array(frames)  # (T, H, W, C)
```

**3.2 차원 재배치**
```python
arr = arr.transpose(3, 0, 1, 2)  # (C, T, H, W)
```
- **이유**: PyTorch 모델은 채널 우선 형식 사용

**3.3 데이터 타입 변환 및 정규화**
```python
if arr.dtype != np.float32:
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
```
- **정규화**: [0, 255] → [0, 1] 범위로 변환
- **타입**: float32 (모델 입력 요구사항)

**3.4 파일 저장**
```python
npy_path = f"tmp_npy/{uuid.uuid4().hex}.npy"
np.save(npy_path, arr)
```

#### 출력
- **파일 경로**: `tmp_npy/{uuid}.npy`
- **데이터 형식**: `(C=3, T=40, H=112, W=112)` float32 배열
- **값 범위**: [0.0, 1.0]

#### 참고
- 프레임 수는 가변적이지만, 모델은 40프레임을 기대
- 실제 구현에서는 동적 패딩/자르기 수행 가능

---

### 4단계: 의도 예측 (Intent Prediction)

#### 목적
입모양 영상에서 사용자의 의도를 멀티라벨 분류로 예측

#### 처리 과정

**4.1 데이터 로드 및 전처리**
```python
arr = np.load(npy_path)
# 타입 및 정규화 재확인
if arr.dtype != np.float32:
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
```

**4.2 Tensor 변환**
```python
x = torch.from_numpy(arr.copy()).float()
x = x.unsqueeze(0)  # (C, T, H, W) → (1, C, T, H, W)
x = x.to(device)    # CPU 또는 CUDA
```

**4.3 모델 추론**
```python
logits = model(x)  # TinyLipIntentNet
probs = torch.sigmoid(logits)[0].cpu().numpy()
```

**모델 구조**:
```
입력: (1, 3, 40, 112, 112)
  ↓
[3D CNN Feature Extractor]
  ├─ Conv3D(3→32) + BN + ReLU + MaxPool
  ├─ Conv3D(32→64) + BN + ReLU + MaxPool
  └─ Conv3D(64→128) + BN + ReLU + MaxPool
  ↓
[Transformer Encoder]
  ├─ 2 layers, d_model=128, nhead=4
  └─ 시간적 의존성 모델링
  ↓
[Classifier]
  ├─ Linear(128→128) + ReLU + Dropout(0.3)
  └─ Linear(128→18) → Sigmoid
  ↓
출력: (18,) 확률 점수
```

**4.4 의도 선택**
```python
# 방법 1: Threshold 기반
idxs = np.where(probs >= threshold)[0]  # threshold = 0.3

# 방법 2: Top-K (Threshold 미달 시)
if len(idxs) == 0:
    idxs = np.argsort(probs)[-top_k:]  # top_k = 3

# 정렬 및 제한
idxs = idxs[np.argsort(probs[idxs])[::-1][:top_k]]
```

**4.5 결과 포맷팅**
```python
results = []
for idx in idxs:
    results.append({
        "intent": CANONICAL_KEYWORDS[idx],
        "score": float(probs[idx])
    })
```

#### 출력
```python
[
    {"intent": "TRAVEL", "score": 0.95},
    {"intent": "TIME", "score": 0.78},
    {"intent": "PLACE_CITY", "score": 0.65}
]
```

#### 파라미터
- **top_k**: 3 (최대 선택 의도 개수)
- **threshold**: 0.3 (최소 확률 점수)
- **device**: "cuda" 또는 "cpu" (자동 감지)

---

### 5단계: 문장 생성 (Sentence Generation)

#### 목적
예측된 의도 태그를 자연스러운 한글 문장으로 변환

#### 처리 과정

**5.1 의도별 문장 생성**
```python
for intent_item in intent_list:
    intent_tag = intent_item["intent"]  # "TRAVEL"
    intent_korean = intent_to_korean.get(intent_tag, intent_tag)  # "여행"
```

**5.2 프롬프트 생성**
```python
prompt = f"""
입모양 기반 의도 태그를 자연스러운 한 문장 존댓말로 바꿔주세요.
의도 태그: {intent_tag} ({intent_korean})

규칙:
1) 새로운 정보 추가 금지
2) 태그 의미 안에서만 생성
3) 한 문장으로 자연스럽게 표현
4) 존댓말 사용
5) "~에 대해 말씀하시는 것 같습니다" 같은 단순한 패턴보다는 구체적이고 자연스러운 표현 사용

출력: 문장 하나만 출력하세요.
"""
```

**5.3 Gemini API 호출**

**모델 우선순위**:
```python
models_to_try = [
    "models/gemini-2.5-flash",      # 최신 Flash 모델
    "models/gemini-flash-latest",   # 최신 Flash (자동 업데이트)
    "models/gemini-2.0-flash",      # 2.0 Flash
    "models/gemini-2.5-pro",        # 최신 Pro 모델
    "models/gemini-pro-latest"      # 최신 Pro (자동 업데이트)
]
```

**호출 과정**:
```python
for model_name in models_to_try:
    try:
        model = genai.GenerativeModel(model_name)
        res = model.generate_content(prompt)
        if res and res.text:
            sentence = res.text.strip()
            # 번호나 불릿 포인트 제거
            sentence = sentence.lstrip('0123456789.-•)').strip()
            if sentence and len(sentence) > 5:
                sentences.append(sentence)
                break
    except Exception as e:
        # 할당량 초과가 아니면 다음 모델 시도
        if "429" not in str(e) and "quota" not in str(e).lower():
            continue
        break
```

**5.4 Fallback 메커니즘**
```python
if not sentence_generated:
    fallback_sentence = generate_single_fallback_sentence(
        intent_tag, intent_korean
    )
    sentences.append(fallback_sentence)
```

**Fallback 패턴 예시**:
```python
patterns = {
    "TRAVEL": f"{intent_korean}에 대해 말씀하시는 것 같습니다.",
    "SCHEDULE": f"{intent_korean}에 관해 이야기하시는 것 같습니다.",
    "EMOTION_POS": f"{intent_korean}을 표현하시는 것 같습니다.",
    # ...
}
```

#### 출력
```python
[
    "여행을 계획하고 있습니다.",
    "시간에 대해 말씀하시는 것 같습니다.",
    "도시에 관해 이야기하시는 것 같습니다."
]
```

#### 에러 처리
- API 할당량 초과 (429) → Fallback 사용
- API 연결 실패 → Fallback 사용
- 빈 응답 → Fallback 사용

---

### 6단계: TTS 생성 (Text-to-Speech)

#### 목적
생성된 문장을 한국어 음성으로 변환

#### 처리 과정

**6.1 기본 문장 선택**
```python
default_sentence = sentences[0] if sentences else "의도가 검출되지 않았습니다."
```

**6.2 TTS 생성**
```python
os.makedirs("tts_outputs", exist_ok=True)
out_path = f"tts_outputs/{uuid.uuid4().hex}.mp3"
gTTS(text=default_sentence, lang="ko").save(out_path)
```

**gTTS 설정**:
- **언어**: 한국어 (`lang="ko"`)
- **형식**: MP3
- **파일명**: UUID 기반 고유 이름

#### 출력
- **파일 경로**: `tts_outputs/{uuid}.mp3`
- **형식**: MP3 오디오 파일
- **언어**: 한국어

#### 참고
- 사용자는 웹 UI에서 다른 문장을 선택하여 TTS 생성 가능
- `/api/generate_tts` 엔드포인트 사용

---

## 데이터 형식

### 입력 데이터

#### 비디오 파일
- **형식**: MP4
- **해상도**: 가변 (입 영역만 추출)
- **프레임 레이트**: 가변
- **색상 공간**: RGB

#### Base64 인코딩 (Gradio)
```
data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQAA...
```

### 중간 데이터

#### NumPy 배열
```python
shape: (C=3, T=40, H=112, W=112)
dtype: float32
range: [0.0, 1.0]
```

#### 프레임 리스트
```python
List[np.ndarray]
각 요소: (H, W, 3) uint8 배열
```

### 출력 데이터

#### 의도 리스트
```python
[
    {
        "intent": str,      # "TRAVEL", "TIME", etc.
        "score": float      # 0.0 ~ 1.0
    },
    ...
]
```

#### 문장 리스트
```python
[
    "여행을 계획하고 있습니다.",
    "시간에 대해 말씀하시는 것 같습니다.",
    ...
]
```

#### 오디오 파일
- **형식**: MP3
- **코덱**: MPEG Audio Layer 3
- **샘플 레이트**: 22.05 kHz (gTTS 기본값)
- **비트레이트**: 가변

---

## 에러 처리

### 단계별 에러 처리 전략

#### 1단계: 비디오 입력
- ✅ 비어있는 데이터 → 명확한 에러 메시지
- ✅ 지원하지 않는 타입 → 타입 정보 포함 에러

#### 2단계: 전처리
- ✅ MediaPipe 초기화 실패 → 최소 옵션으로 재시도
- ✅ 얼굴 감지 실패 → `None` 반환, 상위에서 처리
- ✅ 입 박스 계산 실패 → `None` 반환

#### 3단계: NumPy 변환
- ✅ 타입 불일치 → 자동 변환
- ✅ 값 범위 초과 → 자동 정규화

#### 4단계: 의도 예측
- ✅ 모델 로드 실패 → 프로그램 시작 시 확인
- ✅ 추론 실패 → 예외 처리 및 에러 메시지

#### 5단계: 문장 생성
- ✅ API 할당량 초과 → Fallback 사용
- ✅ API 연결 실패 → Fallback 사용
- ✅ 빈 응답 → Fallback 사용
- ✅ 모든 모델 실패 → Fallback 사용

#### 6단계: TTS 생성
- ✅ 디렉토리 없음 → 자동 생성
- ✅ 파일 저장 실패 → 예외 처리

### 전체 파이프라인 에러 처리

```python
try:
    # 전체 파이프라인 실행
    intents, sentences, audio_path = full_pipeline(video_input)
except Exception as e:
    import traceback
    traceback.print_exc()
    return f"에러 발생: {e}", "", ""
```

---

## 성능 최적화

### 1. 멀티프레임 전략
- **목적**: 얼굴 감지 성공률 향상
- **방법**: 5개 위치에서 순차적으로 시도
- **효과**: 초반/후반 프레임 문제 해결

### 2. MediaPipe Fallback
- **목적**: 다양한 환경에서 동작 보장
- **방법**: 초기화 실패 시 최소 옵션으로 재시도
- **효과**: 호환성 향상

### 3. 모델 추론 최적화
- **GPU 사용**: CUDA 자동 감지 및 사용
- **배치 처리**: 향후 확장 가능
- **메모리 관리**: 불필요한 데이터 즉시 해제

### 4. API 호출 최적화
- **모델 우선순위**: 빠른 모델부터 시도
- **Fallback**: API 실패 시 즉시 대체
- **에러 분류**: 할당량 초과와 일반 오류 구분

### 5. 파일 관리
- **UUID 기반**: 파일명 충돌 방지
- **임시 파일**: 처리 후 정리 가능
- **디렉토리 구조**: 체계적인 파일 관리

---

## 사용 예시

### Python에서 직접 호출
```python
from ai_setence_tts_app import full_pipeline

# 파일 경로로 호출
video_path = "test_video.mp4"
intents, sentences, audio_path = full_pipeline(video_path)

print("의도:", intents)
print("문장:", sentences)
print("음성:", audio_path)
```

### FastAPI에서 사용
```python
from ai_setence_tts_app import full_pipeline

@app.post("/api/upload")
async def upload_video(file: UploadFile):
    # 파일 저장
    temp_path = f"uploaded_videos/{uuid.uuid4().hex}.mp4"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    # 파이프라인 실행
    intents, sentences, audio_path = full_pipeline(temp_path)
    
    return {
        "intents": intents,
        "sentences": sentences,
        "audio_url": f"/tts_outputs/{os.path.basename(audio_path)}"
    }
```

---

## 참고 사항

### 필수 파일
- `tiny_lip_intent_best.pth`: 사전 학습된 모델 가중치
- `.env`: `GOOGLE_API_KEY` 설정 필요

### 디렉토리 구조
```
AI_BE/
├── uploaded_videos/    # 입력 영상
├── tmp_video/         # 크롭된 영상
├── tmp_npy/           # NumPy 배열
└── tts_outputs/       # 생성된 음성
```

### 성능 지표
- **전처리**: 약 1-3초 (영상 길이에 따라 가변)
- **의도 예측**: 약 0.1-0.5초 (GPU 사용 시)
- **문장 생성**: 약 1-3초 (API 응답 시간)
- **TTS 생성**: 약 0.5-1초

### 제한 사항
- **영상 형식**: MP4 권장
- **얼굴 감지**: 정면 얼굴 필요
- **의도 개수**: 최대 18개 (현재 설정)
- **API 할당량**: Gemini API 할당량 제한

---

## 문제 해결

### 얼굴 감지 실패
- **원인**: 얼굴이 보이지 않거나 각도 문제
- **해결**: 정면 얼굴이 잘 보이는 영상 사용

### 의도 예측 정확도 낮음
- **원인**: 학습 데이터와 다른 조건
- **해결**: 모델 재학습 또는 데이터 증강

### API 할당량 초과
- **원인**: 너무 많은 요청
- **해결**: Fallback 메커니즘 사용 또는 할당량 증가

---

## 향후 개선 사항

1. **프레임 수 동적 처리**: 가변 길이 영상 지원
2. **배치 처리**: 여러 영상 동시 처리
3. **캐싱**: 동일 영상 재처리 방지
4. **스트리밍**: 실시간 처리 지원
5. **모델 최적화**: 경량화 및 속도 향상

---

**작성일**: 2025년  
**버전**: 1.0  
**작성자**: WHISPEECH Team

