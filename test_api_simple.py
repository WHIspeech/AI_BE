#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""간단한 API 키 테스트"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("✗ API 키가 설정되지 않았습니다!")
    exit(1)

print("API 키 테스트 중...")
print(f"API 키: {api_key[:10]}...{api_key[-4:]}")
print()

try:
    genai.configure(api_key=api_key)
    
    # 가장 간단한 모델로 테스트
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    response = model.generate_content("안녕하세요")
    
    print("✓ API 키 정상 작동!")
    print(f"응답: {response.text}")
    
except Exception as e:
    error_str = str(e)
    if "429" in error_str or "quota" in error_str.lower():
        print("✗ API 할당량 초과")
        print("  → 할당량이 초과되었습니다. 시간이 지난 후 다시 시도하세요.")
    elif "404" in error_str:
        print("✗ 모델을 찾을 수 없음")
        print(f"  오류: {error_str[:100]}")
    else:
        print(f"✗ 오류 발생: {error_str[:200]}")

