import torch
import numpy as np
import pandas as pd
import ast
import faiss
from transformers import AutoTokenizer, AutoModel

#Load Pretrained DistilBERT Model for Feature Extraction
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#Define Emotion Labels
EMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "anticipation", "calm", "caring", "confusion", "curiosity",
    "disappointment", "disgust", "embarrassment", "excitement", "frustration", "gratitude", "grief", "joy", "love",
    "nervousness", "nostalgia", "optimism", "pride", "realization", "relief", "sadness", "satisfaction",
    "surprise", "trust"
]

# 🔹 Load Test Dataset
df = pd.read_csv("C:\Users\Admin\Downloads\PSGTECH\TCS\SURVEY_SPARROW_25\project\CustEmotiTrack\datasets\test_emotion.csv", header=None, names=["Review", "Emotions"])
df = df.sample(frac=0.7, random_state=42).reset_index(drop=True)

# 🔹 Convert Emotion Column (String → List)
def safe_parse_emotions(value):
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)  # Convert string to list
        except (ValueError, SyntaxError):
            return []
    return []

df["Emotions"] = df["Emotions"].apply(safe_parse_emotions)

# 🔹 Convert Labels to Multi-Hot Encoding
def multilabel_binarize(labels, classes):
    return [1 if cls in labels else 0 for cls in classes]

df["labels"] = df["Emotions"].apply(lambda x: multilabel_binarize(x, EMOTIONS))
test_texts = df["Review"].tolist()
test_labels = np.array(df["labels"].tolist())

# 🔹 Function to Get DistilBERT Sentence Embeddings
def get_embedding(text):
    tokens = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    tokens = {key: val.to(device) for key, val in tokens.items()}

    with torch.no_grad():
        outputs = model(**tokens)

    cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Use [CLS] token embedding
    return cls_embedding

# 🔹 Compute Embeddings for All Test Samples
embeddings = np.vstack([get_embedding(text) for text in test_texts])

# 🔹 Build FAISS Index
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)  # L2 distance search
index.add(embeddings)

# 🔹 Fast Emotion Prediction with FAISS
def predict_emotions_faiss(text):
    text_embedding = get_embedding(text)
    _, closest_indices = index.search(text_embedding, k=5)  # Find top 5 closest matches

    # Aggregate Labels of Closest Matches
    predicted_labels = np.mean(test_labels[closest_indices], axis=1).round().astype(int)
    return predicted_labels[0]

# 🔹 Run Predictions for All Samples
predictions = np.array([predict_emotions_faiss(text) for text in test_texts])

# 🔹 Compute Evaluation Metrics
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predictions, average="micro", zero_division=0)

# 🔹 Compute AUC-ROC Score (If Possible)
try:
    auc_roc = roc_auc_score(test_labels, predictions, average="micro")
except ValueError:
    auc_roc = "⚠️ Could not be calculated (check labels)"

# 🔹 Print Evaluation Results
print("\n📊 **Evaluation Metrics:**")
print(f"🔹 Precision: {precision:.4f}")
print(f"🔹 Recall: {recall:.4f}")
print(f"🔹 F1 Score: {f1:.4f}")
print(f"🔹 AUC-ROC Score: {auc_roc}")
