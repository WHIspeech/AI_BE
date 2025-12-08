@echo off
echo ========================================
echo WHISPEECH FastAPI 서버 실행 중...
echo Python 3.10 사용
echo ========================================
echo.

cd /d %~dp0
C:\Users\esthe\AppData\Local\Programs\Python\Python310\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

pause

