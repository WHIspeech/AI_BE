#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MediaPipe 테스트 스크립트"""

import cv2
import numpy as np
import mediapipe as mp

print("=" * 60)
print("MediaPipe FaceMesh 테스트")
print("=" * 60)
print()

# 더미 이미지 생성 (검은색 이미지)
test_image = np.zeros((480, 640, 3), dtype=np.uint8)

mp_face_mesh = mp.solutions.face_mesh

print("FaceMesh 초기화 테스트...")
try:
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        print("✓ FaceMesh 초기화 성공!")
        
        # 이미지 처리 테스트
        rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)
        
        print("✓ 이미지 처리 성공!")
        print(f"  얼굴 검출: {results.multi_face_landmarks is not None}")
        
except Exception as e:
    print(f"✗ 오류 발생: {str(e)}")
    print()
    print("가능한 해결 방법:")
    print("1. MediaPipe 버전을 업데이트: pip install --upgrade mediapipe")
    print("2. MediaPipe 버전을 다운그레이드: pip install mediapipe==0.10.0")
    exit(1)

print()
print("=" * 60)
print("✓ MediaPipe가 정상적으로 작동합니다!")
print("=" * 60)

