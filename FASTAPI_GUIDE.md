# FastAPI 웹 애플리케이션 사용 가이드

## 개요
WHISPEECH 프로젝트를 FastAPI 기반 웹 애플리케이션으로 변환했습니다.

## 파일 구조
```
AI_BE/
├── app.py                 # FastAPI 메인 애플리케이션
├── static/
│   └── index.html        # 프론트엔드 HTML 페이지
├── run_fastapi.bat       # FastAPI 서버 실행 스크립트
└── requirements.txt      # 필요한 패키지 목록
```

## 설치 및 실행

### 1. 패키지 설치
```bash
C:\Users\esthe\AppData\Local\Programs\Python\Python310\python.exe -m pip install fastapi uvicorn[standard] python-multipart
```

또는
```bash
C:\Users\esthe\AppData\Local\Programs\Python\Python310\python.exe -m pip install -r requirements.txt
```

### 2. 환경 변수 설정
`.env` 파일을 `AI_BE` 폴더에 생성하고 다음 내용 추가:
```
GOOGLE_API_KEY=여기에_API_키_입력
```

### 3. 서버 실행

**방법 1: 배치 파일 사용 (권장)**
```cmd
cd C:\whispeech\AI_BE
run_fastapi.bat
```

**방법 2: 직접 실행**
```cmd
cd C:\whispeech\AI_BE
C:\Users\esthe\AppData\Local\Programs\Python\Python310\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. 웹 브라우저에서 접속
서버 실행 후 브라우저에서 다음 URL로 접속:
```
http://localhost:8000
```

## API 엔드포인트

### 1. 메인 페이지
- **GET** `/`
- HTML 프론트엔드 페이지 반환

### 2. 영상 업로드 및 처리
- **POST** `/api/upload`
- **요청**: `multipart/form-data` 형식으로 MP4 파일 업로드
- **응답**: JSON 형식
  ```json
  {
    "success": true,
    "intents": [
      {"intent": "TRAVEL", "score": 0.95},
      {"intent": "TIME", "score": 0.78}
    ],
    "sentence": "여행을 계획하고 있습니다.",
    "audio_url": "/tts_outputs/abc123.mp3",
    "file_id": "abc123"
  }
  ```

### 3. 헬스 체크
- **GET** `/api/health`
- 서버 상태 확인

### 4. 오디오 파일 다운로드
- **GET** `/tts/{filename}`
- 생성된 TTS 오디오 파일 다운로드

## 사용 방법

1. 웹 브라우저에서 `http://localhost:8000` 접속
2. 입모양이 잘 보이는 MP4 영상 파일 업로드
   - 드래그 앤 드롭 또는 클릭하여 파일 선택
3. "복원하기" 버튼 클릭
4. 결과 확인:
   - **예측된 의도**: 검출된 의도 태그와 신뢰도
   - **복원된 문장**: Gemini가 생성한 자연어 문장
   - **생성된 음성**: TTS로 생성된 오디오 재생

## 특징

- ✅ 모던한 웹 UI (드래그 앤 드롭 지원)
- ✅ 실시간 처리 상태 표시
- ✅ RESTful API 구조
- ✅ CORS 지원 (다른 도메인에서도 접근 가능)
- ✅ 오류 처리 및 사용자 친화적 메시지

## 문제 해결

### 포트가 이미 사용 중인 경우
다른 포트로 실행:
```bash
uvicorn app:app --host 0.0.0.0 --port 8001
```

### 모듈을 찾을 수 없는 오류
Python 3.10으로 실행 중인지 확인:
```bash
C:\Users\esthe\AppData\Local\Programs\Python\Python310\python.exe --version
```

### API 키 오류
`.env` 파일이 올바른 위치에 있고 API 키가 정확한지 확인

