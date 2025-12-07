import google.generativeai as genai

def generate_sentence_from_intents(intents):
    if not intents:
        return "입모양에서 의도를 찾지 못했습니다."

    tags = [x["intent"] for x in intents]

    prompt = f"""
입모양 의도 {tags} 를 한 문장 존댓말로 바꾸세요.
"""

    model = genai.GenerativeModel("gemini-1.5-flash")
    res = model.generate_content(prompt)

    return res.text.strip()
