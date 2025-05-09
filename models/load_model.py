import torch
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

# ✅ Load Model Directly from Hugging Face
MODEL_NAME = "joeddav/distilbert-base-uncased-go-emotions-student"

# ✅ Load Tokenizer and Model from Hugging Face (No Local Download)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# ✅ Move Model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("✅ Model and Tokenizer Loaded Successfully from Hugging Face!")

# ✅ Emotion Labels
EMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire",
    "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy",
    "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

# ✅ Activation Levels Mapping
def get_activation_level(intensity):
    if intensity >= 0.8:
        return "High"
    elif intensity >= 0.3:
        return "Medium"
    else:
        return "Low"

# ✅ Prediction Function
def predict_emotions(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits.squeeze(0).cpu().numpy()  # Extract logits and move to CPU
    scores = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy().flatten()

    # Normalize logits for intensity calculation
    min_logit, max_logit = logits.min(), logits.max()
    if max_logit == min_logit:  # Prevent division by zero
        intensity_scores = [0] * len(logits)
    else:
        intensity_scores = [(logits[i] - min_logit) / (max_logit - min_logit) for i in range(len(logits))]

    # Convert to JSON format
    sentiment_scores = {EMOTIONS[i]: float(scores[i]) for i in range(len(EMOTIONS))}
    sorted_emotions = sorted(sentiment_scores.items(), key=lambda x: x[1], reverse=True)

    # Select Top 2 Emotions
    top_2 = sorted_emotions[:2]
    idx_1 = EMOTIONS.index(top_2[0][0])
    idx_2 = EMOTIONS.index(top_2[1][0])

    result = {
        "emotions": {
            "primary": {
                "emotion": top_2[0][0],
                "activation": get_activation_level(intensity_scores[idx_1]),  # Correct index lookup
                "intensity": round(float(intensity_scores[idx_1]), 2),
                "confidence": round(float(top_2[0][1]), 2)  # Confidence from softmax
            },
            "secondary": {
                "emotion": top_2[1][0],
                "activation": get_activation_level(intensity_scores[idx_2]),  # Correct index lookup
                "intensity": round(float(intensity_scores[idx_2]), 2),
                "confidence": round(float(top_2[1][1]), 2)
            }
        },
        "adorescore": {
            "overall": round((top_2[0][1] + top_2[1][1]) * 50, 2)  # Adorescore formula
        }
    }

    return result
