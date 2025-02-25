import torch
import numpy as np
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load Pretrained DistilBERT Model for Emotion Detection
# ✅ Use the model designed for multi-label emotion classification:
MODEL_NAME = "joeddav/distilbert-base-uncased-go-emotions-student"  
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define Emotion Labels
EMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire",
    "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy",
    "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

# Define Positive & Negative Emotion Sets for Adorescore
POSITIVE_EMOTIONS = {"admiration", "amusement", "approval", "caring", "excitement", "gratitude", "joy", "love", "optimism", "pride", "relief"}
NEGATIVE_EMOTIONS = {"anger", "annoyance", "disappointment", "disapproval", "disgust", "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness"}

# Function to Predict Emotion Scores
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.sigmoid(outputs.logits).cpu().numpy().flatten()  # ✅ Apply sigmoid for multi-label
    return {EMOTIONS[i]: scores[i] for i in range(len(EMOTIONS))}

# Function to Get Primary & Secondary Emotions
def get_primary_secondary_emotions(sentiment_scores):
    sorted_emotions = sorted(sentiment_scores.items(), key=lambda x: x[1], reverse=True)
    primary = {"emotion": sorted_emotions[0][0], "activation": "Medium", "intensity": round(sorted_emotions[0][1], 2)}
    secondary = {"emotion": sorted_emotions[1][0], "activation": "Low", "intensity": round(sorted_emotions[1][1], 2)}
    return primary, secondary

# Function to Calculate Adorescore
def calculate_adorescore(sentiment_scores):
    pos_sum = sum(sentiment_scores[e] for e in POSITIVE_EMOTIONS if e in sentiment_scores)
    neg_sum = sum(sentiment_scores[e] for e in NEGATIVE_EMOTIONS if e in sentiment_scores)
    adorescore = 100 * (pos_sum - neg_sum) / (pos_sum + neg_sum + 1e-6)
    return round(adorescore, 2)

# Function to Get Emotion JSON Output
def get_emotion_analysis(text):
    sentiment_scores = predict_sentiment(text)
    primary_emotion, secondary_emotion = get_primary_secondary_emotions(sentiment_scores)
    adorescore = calculate_adorescore(sentiment_scores)

    # Output JSON
    output_json = {
        "emotions": {
            "primary": primary_emotion,
            "secondary": secondary_emotion
        },
        "adorescore": {
            "overall": adorescore
        }
    }
    return output_json

# Example Usage
if __name__ == "__main__":
    review = "Fast delivery and great packaging. Highly recommend!"
    result = get_emotion_analysis(review)
    print(json.dumps(result, indent=4, default=float))