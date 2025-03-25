import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ✅ Define Save Paths
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_NAME = "joeddav/distilbert-base-uncased-go-emotions-student"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

print("✅ Model Loaded")

# ✅ Save Locally
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model.pt")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer")

torch.save(model.state_dict(), MODEL_PATH)
tokenizer.save_pretrained(TOKENIZER_PATH)

print(f"💾 Model saved at: {MODEL_PATH}")
print(f"💾 Tokenizer saved at: {TOKENIZER_PATH}")
