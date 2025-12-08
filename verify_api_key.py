#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""API 키 확인 스크립트"""

import os
from dotenv import load_dotenv

print("=" * 60)
print("API 키 확인")
print("=" * 60)
print()

# .env 파일 직접 읽기
env_path = ".env"
if os.path.exists(env_path):
    print("✓ .env 파일 존재")
    with open(env_path, 'r', encoding='utf-8') as f:
        content = f.read()
        print("\n.env 파일 내용:")
        print("-" * 60)
        for line in content.split('\n'):
            if line.strip() and not line.strip().startswith('#'):
                if 'GOOGLE_API_KEY' in line:
                    # API 키 마스킹
                    if '=' in line:
                        key_part = line.split('=')[1].strip()
                        if key_part:
                            masked = key_part[:10] + "..." + key_part[-4:] if len(key_part) > 14 else "***"
                            print(f"{line.split('=')[0]}={masked}")
                        else:
                            print(line)
                    else:
                        print(line)
                else:
                    print(line)
        print("-" * 60)
else:
    print("✗ .env 파일이 없습니다!")
    exit(1)

print()

# dotenv로 로드된 값 확인
load_dotenv()
loaded_key = os.getenv("GOOGLE_API_KEY")

if loaded_key:
    masked_loaded = loaded_key[:10] + "..." + loaded_key[-4:] if len(loaded_key) > 14 else "***"
    print(f"✓ dotenv로 로드된 API 키: {masked_loaded}")
    print(f"  전체 길이: {len(loaded_key)} 문자")
else:
    print("✗ dotenv로 API 키를 로드할 수 없습니다!")

print()
print("=" * 60)

