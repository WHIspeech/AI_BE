#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""API 연동 상태 확인 스크립트"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

print("=" * 60)
print("API 연동 상태 확인")
print("=" * 60)
print()

# 1. .env 파일 확인
env_path = ".env"
if os.path.exists(env_path):
    print("✓ .env 파일이 존재합니다.")
    load_dotenv()
else:
    print("✗ .env 파일이 없습니다!")
    print("  → AI_BE 폴더에 .env 파일을 생성하고 다음 내용을 추가하세요:")
    print("     GOOGLE_API_KEY=여기에_API_키_입력")
    print()
    exit(1)

print()

# 2. API 키 확인
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    # API 키가 있는 경우 마스킹해서 표시
    masked_key = api_key[:10] + "..." + api_key[-4:] if len(api_key) > 14 else "***"
    print(f"✓ GOOGLE_API_KEY가 설정되어 있습니다. ({masked_key})")
else:
    print("✗ GOOGLE_API_KEY가 설정되지 않았습니다!")
    print("  → .env 파일에 다음 내용을 추가하세요:")
    print("     GOOGLE_API_KEY=여기에_API_키_입력")
    print()
    print("  API 키 발급: https://makersuite.google.com/app/apikey")
    exit(1)

print()

# 3. Gemini API 연결 테스트
print("Gemini API 연결 테스트 중...")
try:
    genai.configure(api_key=api_key)
    
    # 사용 가능한 최신 모델 찾기
    models_to_try = [
        "models/gemini-2.5-flash",
        "models/gemini-flash-latest",
        "models/gemini-2.0-flash",
        "models/gemini-2.5-pro",
        "models/gemini-pro-latest"
    ]
    model = None
    model_name = None
    response = None
    
    for m_name in models_to_try:
        try:
            model = genai.GenerativeModel(m_name)
            # 간단한 테스트 요청
            response = model.generate_content("안녕하세요")
            if response and response.text:
                model_name = m_name
                break
        except Exception as e:
            print(f"  모델 {m_name} 시도 실패: {str(e)[:50]}...")
            continue
    
    if not model or not response:
        print("✗ 사용 가능한 Gemini 모델을 찾을 수 없습니다.")
        print("  사용 가능한 모델 목록을 확인하세요.")
        exit(1)
    
    # 성공한 모델로 테스트
    print(f"  사용 모델: {model_name}")
    
    if response and response.text:
        print("✓ Gemini API 연결 성공!")
        print(f"  테스트 응답: {response.text[:50]}...")
    else:
        print("✗ Gemini API 응답이 비어있습니다.")
        exit(1)
        
except Exception as e:
    print(f"✗ Gemini API 연결 실패!")
    print(f"  오류: {str(e)}")
    print()
    print("  가능한 원인:")
    print("  1. API 키가 잘못되었습니다.")
    print("  2. 인터넷 연결이 없습니다.")
    print("  3. API 할당량이 초과되었습니다.")
    exit(1)

print()
print("=" * 60)
print("✓ 모든 API 연동이 정상적으로 설정되어 있습니다!")
print("=" * 60)

