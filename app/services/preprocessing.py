import cv2
import mediapipe as mp
import numpy as np
import uuid
import os

mp_face_mesh = mp.solutions.face_mesh

LIP_LANDMARKS = [61,146,91,181,84,17,314,405,321,375,291,
                 78,95,88,178,87,14,317,402,318,324,308]


def find_best_mouth_frame(cap, total_frames):
    check_positions = [0.05, 0.25, 0.5, 0.75, 0.95]
    check_frames = [int(total_frames * p) for p in check_positions]

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        for idx in check_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                return frame
    return None


def get_mouth_box(frame, face_mesh, pad_x=1.2, pad_y=1.8):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0].landmark

    xs = [lm[idx].x * w for idx in LIP_LANDMARKS]
    ys = [lm[idx].y * h for idx in LIP_LANDMARKS]

    min_x, max_x = int(min(xs)), int(max(xs))
    min_y, max_y = int(min(ys)), int(max(ys))

    return min_x, min_y, max_x, max_y


def preprocess_video(input_path):
    """영상에서 입 영역을 crop하여 모든 frame을 리스트로 반환"""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    best_frame = find_best_mouth_frame(cap, total)

    if best_frame is None:
        return None

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        box = get_mouth_box(best_frame, face_mesh)
        if box is None:
            return None

        x1, y1, x2, y2 = box

    # crop all frames
    cap = cv2.VideoCapture(input_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        crop = frame[y1:y2, x1:x2]
        frames.append(crop)

    cap.release()

    return frames


def frames_to_npy(frames):
    os.makedirs("app/static/tmp", exist_ok=True)
    npy_path = f"app/static/tmp/{uuid.uuid4().hex}.npy"
    np.save(npy_path, np.array(frames))
    return npy_path
