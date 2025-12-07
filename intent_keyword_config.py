# intent_keyword_config.py

# 표준화된 intent 이름
CANONICAL_KEYWORDS = [
    # 여행 도메인
    "TRAVEL",
    "SCHEDULE",
    "PLACE_NATURE",
    "PLACE_CITY",
    "COMPANION",
    "FOOD_DRINK",
    "TRANSPORT",
    "COST",
    "TIME",
    "ACTIVITY",
    "EMOTION_POS",
    "EMOTION_NEG",

    # 케어(환자) 도메인 – 향후 실제 서비스용
    "WATER",
    "FOOD_CARE",
    "TOILET",
    "PAIN",
    "HELP",
    "MOVE",
]
# 각 intent와 매핑되는 키워드들
TRIGGER_MAP = {
    "TRAVEL": ["여행", "해외여행", "관광지", "여행지", "휴가", ...],
    "TIME":   ["이번", "주말", "작년", "내년", "올해", "언제", ...],
    "PLACE_CITY": ["도시", "시내", "시장", "호텔", "숙소", "리조트", ...],
    "WATER": ["물 주세요", "물 한 잔", "물 좀", "목마르다", "목말라", ...],
    "HELP":  ["도와줘", "도와주세요", "살려줘", "위험해", "응급", "도움이 필요", ...],
}