import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp

# ============================================================
# 0) Mediapipe 설정
# ============================================================
mp_face_mesh = mp.solutions.face_mesh

# 입 주변 랜드마크 index
LIP_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308
]


# ============================================================
# 1) TinyLipIntentNet (필수 모델 구조)
# ============================================================
class TinyLipIntentNet(nn.Module):
    def __init__(self, num_intents: int, in_channels: int = 3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3,5,5), padding=(1,2,2)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,2,2)),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2,2,2)),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2,2,2)),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_intents)
        )

    def forward(self, x):
        # x: (B,C,T,H,W)
        feat = self.features(x)
        B, C, T, H, W = feat.shape
        feat = feat.mean(dim=[3,4])     # (B,C,T)
        feat = feat.permute(0,2,1)      # (B,T,C)

        feat = self.transformer(feat)
        feat = feat.mean(dim=1)         # (B,C)

        logits = self.classifier(feat)
        return logits


# ============================================================
# 2) 입 영역 추출 (영상 → crop frames)
# ============================================================
def get_mouth_box(frame, face_mesh, pad_x=0.5, pad_y=0.9):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    xs, ys = [], []
    for idx in LIP_LANDMARKS:
        xs.append(landmarks[idx].x * w)
        ys.append(landmarks[idx].y * h)

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    box_w = max_x - min_x
    box_h = max_y - min_y

    min_x = int(max(0, min_x - box_w * pad_x))
    max_x = int(min(w, max_x + box_w * pad_x))
    min_y = int(max(0, min_y - box_h * pad_y))
    max_y = int(min(h, max_y + box_h * pad_y))

    return min_x, min_y, max_x, max_y


# ============================================================
# 3) 영상 → (C,T,H,W) npy 변환
# ============================================================
def video_to_npy(video_path, target_T=40, size=(112,112)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"영상 열기 실패: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_idx = frame_count // 2

    # ① 대표 프레임에서 입 위치 찾기
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        mid_frame = None
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            if i == mid_idx:
                mid_frame = frame.copy()
                break

        if mid_frame is None:
            raise RuntimeError("중간 프레임 로드 실패")

        box = get_mouth_box(mid_frame, face_mesh)
        if box is None:
            raise RuntimeError("입 영역을 찾지 못했습니다.")

        x1, y1, x2, y2 = box

    # ② 동일 박스로 모든 프레임 자르기
    cap.release()
    cap = cv2.VideoCapture(video_path)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, size)
        frames.append(crop)

    cap.release()

    arr = np.stack(frames, axis=0)  # (T,H,W,C)

    # 채널 먼저 → (C,T,H,W)
    arr = arr.transpose(3,0,1,2)

    # float32 정규화
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr /= 255.0

    # T 길이 40으로 맞춤
    C, T, H, W = arr.shape
    if T > target_T:
        start = (T - target_T) // 2
        arr = arr[:, start:start+target_T]
    else:
        pad = target_T - T
        left = pad // 2
        right = pad - left
        arr = np.pad(arr, ((0,0),(left,right),(0,0),(0,0)), mode="edge")

    return arr  # (C,T,H,W)


# ============================================================
# 4) Intent 예측 (TinyLipIntentNet inference)
# ============================================================
@torch.no_grad()
def predict_intents(model, arr, canonical_keywords, top_k=3, threshold=0.3, device="cpu"):
    """
    arr : (C,T,H,W) numpy array
    """
    x = torch.from_numpy(arr.copy()).unsqueeze(0).to(device)  # (1,C,T,H,W)

    logits = model(x)
    probs = torch.sigmoid(logits)[0].cpu().numpy()  # (num_intents,)

    idxs = np.where(probs >= threshold)[0]
    if len(idxs) == 0:
        idxs = np.argsort(probs)[-top_k:]
    else:
        idxs = idxs[np.argsort(probs[idxs])[::-1][:top_k]]

    results = []
    for idx in idxs:
        results.append({
            "intent": canonical_keywords[idx],
            "score": float(probs[idx])
        })
    return results


# ============================================================
# 5) dummy intent 모드 (weight 없을 때)
# ============================================================
def dummy_intents():
    return [
        {"intent": "TRAVEL", "score": 0.92},
        {"intent": "TIME", "score": 0.55},
    ]
