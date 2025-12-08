#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Gemini 사용 가능한 모델 확인"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

print("=" * 60)
print("사용 가능한 Gemini 모델 확인")
print("=" * 60)
print()

try:
    # 사용 가능한 모델 목록 가져오기
    models = genai.list_models()
    
    print("사용 가능한 모델:")
    for model in models:
        if 'generateContent' in model.supported_generation_methods:
            print(f"  - {model.name}")
            print(f"    지원 메서드: {model.supported_generation_methods}")
            print()
    
    # 테스트할 모델들
    test_models = [
        "gemini-pro",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "models/gemini-pro",
        "models/gemini-1.5-pro",
        "models/gemini-1.5-flash"
    ]
    
    print("\n" + "=" * 60)
    print("모델 테스트")
    print("=" * 60)
    
    for model_name in test_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content("안녕")
            print(f"✓ {model_name}: 성공 - {response.text[:50]}...")
            break
        except Exception as e:
            print(f"✗ {model_name}: 실패 - {str(e)[:80]}...")
    
except Exception as e:
    print(f"오류: {e}")

