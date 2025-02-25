import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ✅ Define Model Name (Pretrained DistilBERT for Emotions)
MODEL_NAME = "joeddav/distilbert-base-uncased-go-emotions-student"

# ✅ Load Pretrained Model & Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# ✅ Move Model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ✅ Save Model & Tokenizer
MODEL_PATH = "models/emotion_model.pt"
TOKENIZER_PATH = "models/tokenizer"

# Save model state_dict (only model weights)
torch.save(model.state_dict(), MODEL_PATH)

# Save tokenizer separately
tokenizer.save_pretrained(TOKENIZER_PATH)

print(f"✅ Model saved at: {MODEL_PATH}")
print(f"✅ Tokenizer saved at: {TOKENIZER_PATH}")
