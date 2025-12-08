# Python 3.10으로 실행하기

## 현재 상황
- Python 3.7.9: 기본 `python` 명령어로 실행됨
- Python 3.10.6: 이미 설치되어 있음 (`C:\Users\esthe\AppData\Local\Programs\Python\Python310\python.exe`)

## Python 3.10으로 실행하는 방법

### 방법 1: 전체 경로 사용 (권장)
```cmd
C:\Users\esthe\AppData\Local\Programs\Python\Python310\python.exe ai_setence_tts_app.py
```

### 방법 2: 배치 파일 생성
`run.bat` 파일을 만들어서 실행:
```batch
@echo off
C:\Users\esthe\AppData\Local\Programs\Python\Python310\python.exe ai_setence_tts_app.py
pause
```

### 방법 3: Python 3.10을 기본으로 설정
1. 환경 변수 PATH에서 Python 3.10 경로를 Python 3.7보다 앞에 배치
2. 또는 `py` 런처 사용:
   ```cmd
   py -3.10 ai_setence_tts_app.py
   ```

## 패키지 설치 확인
```cmd
C:\Users\esthe\AppData\Local\Programs\Python\Python310\python.exe test_python_version.py
```

## 패키지 설치 (필요시)
```cmd
C:\Users\esthe\AppData\Local\Programs\Python\Python310\python.exe -m pip install -r requirements.txt
```

## 환경 변수 설정
`.env` 파일을 `AI_BE` 폴더에 생성하고 다음 내용 추가:
```
GOOGLE_API_KEY=여기에_API_키_입력
```

API 키 발급: https://makersuite.google.com/app/apikey

