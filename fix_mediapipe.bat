@echo off
echo ========================================
echo MediaPipe 다운그레이드
echo ========================================
echo.
echo 주의: 서버가 실행 중이면 먼저 Ctrl+C로 중지하세요!
echo.
pause

echo MediaPipe 0.10.20 설치 중...
C:\Users\esthe\AppData\Local\Programs\Python\Python310\python.exe -m pip install --force-reinstall mediapipe==0.10.20

echo.
echo 완료!
pause

