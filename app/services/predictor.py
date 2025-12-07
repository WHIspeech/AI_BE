import torch
from tiny_lip_intent_model import TinyLipIntentNet, predict_intents
from intent_keyword_config import CANONICAL_KEYWORDS

def load_intent_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_intents = len(CANONICAL_KEYWORDS)

    model = TinyLipIntentNet(num_intents=num_intents, in_channels=3)
    model.load_state_dict(torch.load("tiny_lip_intent_best.pth", map_location="cpu"))
    model.to(device)
    model.eval()

    return model, CANONICAL_KEYWORDS, device
