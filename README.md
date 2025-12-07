# AI_BE
## ⭐ WHISPEECH – Silent Speech Restoration System

입모양만 보고 사용자의 발화를 복원하고, 문맥 기반 자연스러운 문장으로 확장하는 AI 서비스

WHISPEECH는 묵음 상태의 입모양(영상)만으로 사용자의 실제 발화를 예측하는 Silent Speech Restoration 모델입니다.
입모양에서 텍스트를 복원한 뒤, LLM 기반 의도 분석 및 대화 문장 생성까지 수행하여 사용자 커뮤니케이션을 돕는 것을 목표로 합니다.


## 📌 Features (주요 기능)

🔹 1. 묵음 발화 → 의도 추출 모델
- AI Hub 립리딩 영상 기반 **입모양 분석 모델** 개발  
- MediaPipe FaceMesh로 입 주변 ROI 자동 추출  
- Video → Frame → Numpy → 3D CNN + Transformer 기반 **멀티라벨 의도 분류**  

🔹 2. LLM 기반 문장 생성 (의도 → 자연스러운 문장)
- 예측된 의도 태그를 기반으로 LLM(Gemini)을 활용해
- 자연스럽고 문맥 있는 문장으로 변환
- 규칙 기반 프롬프트 설계로 정보 왜곡 최소화

## 🎥 System Architecture(시스템 아키텍쳐)
 Video Upload
        ↓
[Preprocessing Service]
- Frame extraction  
- FaceMesh landmark detection  
- Mouth ROI crop  
- NPY 변환
        ↓
[Intent Model Service]
- TinyLipIntent (3D CNN + Transformer)
- Multi-label classification
        ↓
[Sentence Generator]
- Gemini 1.5 Flash
- Intent → One polite sentence
        ↓
[TTS Service]
- gTTS
- MP3 생성
        ↓
💬 최종 반환 (Intent JSON / 문장 / 음성파일)


## 🛠 Tech Stack

### **AI / Deep Learning**
<p>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg" height="40"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/opencv/opencv-original.svg" height="40"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" height="40"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/mediapipe/mediapipe-original.svg" height="40"/>
</p>

### **LLM / Cloud**
<p>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/googlecloud/googlecloud-original.svg" height="40"/>
</p>

### **Tools**
<p>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" height="40"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/git/git-original.svg" height="40"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/docker/docker-original.svg" height="40"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/vscode/vscode-original.svg" height="40"/>
</p>

### **Frontend**
<p>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/html5/html5-original.svg" height="40" />
</p>

### **Backend**
<p>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" height="40" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/fastapi/fastapi-original.svg" height="40"/>
</p>


🎬 Demo
