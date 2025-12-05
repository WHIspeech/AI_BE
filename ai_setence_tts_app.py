# ai_sentence_tts_app.py
# ====================================================
# Upstage Solar LLM + gTTS + Gradio
# ====================================================

import os
import uuid
import gradio as gr
from gtts import gTTS
import requests

from dotenv import load_dotenv
# ====================================================
# 1) .env 파일 로드
# ====================================================
load_dotenv()  # .env 파일 읽어오기

UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
SOLAR_URL = "https://api.upstage.ai/v1/chat/completions"

if not UPSTAGE_API_KEY:
    raise ValueError("❌ UPSTAGE_API_KEY가 설정되지 않았습니다! .env 파일을 확인하세요.")


def generate_sentence_from_keywords(keyword_list):
    """
    TinyLipNet이 출력한 후보 단어들(keyword_list)을
    자연스러운 한국어 문장으로 보정하는 함수.
    
    ※ 새로운 단어 추가 절대 금지
    ※ 후보 단어 순서 최대한 유지
    ※ 창작 방지 → temperature=0.2
    """

    if isinstance(keyword_list, str):
        kw_list = [k.strip() for k in keyword_list.split(",") if k.strip()]
    else:
        kw_list = [str(k).strip() for k in keyword_list if k]

    if not kw_list:
        return "후보 단어가 없습니다."

    prompt = f"""
당신은 환자의 '입모양 기반'으로 추출된 후보 단어들을 자연스러운 한국어 문장으로 조합하는 전문가입니다.

주의사항(아주 중요):
1) 반드시 아래 '후보 단어들'만 사용하세요.
2) 새로운 단어를 절대 추가하지 마세요.
3) 후보 단어의 의미 범위를 벗어난 내용을 생성하지 마세요.
4) 후보 단어들의 순서를 최대한 유지하세요.
5) 한 문장만 출력하세요.
6) 존댓말을 사용하세요.

후보 단어들: {kw_list}

출력은 문장 1개만 작성하세요.
"""

    response = requests.post(
        SOLAR_URL,
        headers={
            "Authorization": f"Bearer {UPSTAGE_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "solar-1-mini-chat",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,   # 🔒 창작 억제
            "max_tokens": 64
        }
    )

    result = response.json()
    sentence = result["choices"][0]["message"]["content"].strip()
    return sentence

# ====================================================
# 2) 문장 → 음성 (gTTS)
# ====================================================
def generate_tts(sentence):
    if not sentence:
        return None

    os.makedirs("tts_outputs", exist_ok=True)
    filename = f"tts_{uuid.uuid4().hex}.mp3"
    filepath = os.path.join("tts_outputs", filename)

    gTTS(sentence, lang="ko").save(filepath)
    return filepath


# ====================================================
# 3) Gradio 파이프라인
# ====================================================
def run_pipeline(keyword_input):
    sentence = generate_sentence_from_keywords(keyword_input)
    audio_path = generate_tts(sentence)
    return sentence, audio_path


# ====================================================
# 4) Gradio UI
# ====================================================
with gr.Blocks() as demo:
    gr.Markdown("## 🌞 WHISPEECH - Solar 기반 묵음 발화 복원")

    keyword_box = gr.Textbox(label="키워드 입력")
    generate_btn = gr.Button("생성하기")

    out_sentence = gr.Textbox(label="생성된 문장")
    out_audio = gr.Audio(label="생성된 음성", type="filepath")

    generate_btn.click(
        run_pipeline,
        inputs=keyword_box,
        outputs=[out_sentence, out_audio]
    )

if __name__ == "__main__":
    demo.launch()
