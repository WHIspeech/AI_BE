#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Python 버전 및 패키지 확인 스크립트"""

import sys

print("=" * 50)
print("Python 버전 확인")
print("=" * 50)
print(f"Python 버전: {sys.version}")
print(f"Python 실행 경로: {sys.executable}")
print()

print("=" * 50)
print("필수 패키지 확인")
print("=" * 50)

packages = {
    "torch": "torch",
    "gradio": "gradio",
    "mediapipe": "mediapipe",
    "cv2": "opencv-python",
    "gtts": "gtts",
    "dotenv": "python-dotenv",
    "genai": "google-generativeai",
    "numpy": "numpy"
}

for module_name, package_name in packages.items():
    try:
        if module_name == "cv2":
            import cv2
            print(f"✓ {package_name}: {cv2.__version__}")
        elif module_name == "dotenv":
            import dotenv
            print(f"✓ {package_name}: {dotenv.__version__}")
        elif module_name == "genai":
            import google.generativeai as genai
            print(f"✓ {package_name}: 설치됨")
        else:
            module = __import__(module_name)
            version = getattr(module, "__version__", "설치됨")
            print(f"✓ {package_name}: {version}")
    except ImportError as e:
        print(f"✗ {package_name}: 설치되지 않음 - {e}")

print()
print("=" * 50)
if sys.version_info >= (3, 8):
    print("✓ Python 3.8 이상 버전입니다. 정상 작동 가능합니다!")
else:
    print("✗ Python 3.8 미만 버전입니다. 업그레이드가 필요합니다.")
print("=" * 50)

