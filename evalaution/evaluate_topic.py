import torch
import numpy as np
import pandas as pd
import ast
import faiss
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

# 🔹 Load Pretrained DistilBERT Model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 🔹 Load Test Dataset
df = pd.read_csv("datasets/test_topics.csv", header=None, names=["Review", "Topics", "Subtopics"])
df = df.sample(frac=0.7, random_state=42).reset_index(drop=True)

# 🔹 Convert String Representations of Topics/Subtopics to Lists
def safe_parse(value):
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
    return []

df["Topics"] = df["Topics"].apply(safe_parse)
df["Subtopics"] = df["Subtopics"].apply(safe_parse)

# 🔹 Define Topic Labels (Replace with Your Actual Topics)
TOPICS = ["Delivery", "Service", "Product Quality", "Pricing", "Customer Support"]
SUBTOPICS = ["Fast Shipping", "Late Delivery", "Affordable", "Expensive", "Durability", "Response Time"]

# 🔹 Convert Labels to Multi-Hot Encoding
def multilabel_binarize(labels, classes):
    return [1 if cls in labels else 0 for cls in classes]

df["topic_labels"] = df["Topics"].apply(lambda x: multilabel_binarize(x, TOPICS))
df["subtopic_labels"] = df["Subtopics"].apply(lambda x: multilabel_binarize(x, SUBTOPICS))

test_texts = df["Review"].tolist()
test_topic_labels = np.array(df["topic_labels"].tolist())
test_subtopic_labels = np.array(df["subtopic_labels"].tolist())

# 🔹 Function to Get DistilBERT Sentence Embeddings
def get_embedding(text):
    tokens = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    tokens = {key: val.to(device) for key, val in tokens.items()}

    with torch.no_grad():
        outputs = model(**tokens)

    cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return cls_embedding

# 🔹 Compute Embeddings for All Test Samples
embeddings = np.vstack([get_embedding(text) for text in test_texts])

# 🔹 Build FAISS Index
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)

# 🔹 Fast Topic Prediction with FAISS
def predict_topics_faiss(text):
    text_embedding = get_embedding(text)
    _, closest_indices = index.search(text_embedding, k=5)

    # Aggregate Labels of Closest Matches
    predicted_topics = np.mean(test_topic_labels[closest_indices], axis=1).round().astype(int)
    predicted_subtopics = np.mean(test_subtopic_labels[closest_indices], axis=1).round().astype(int)

    return predicted_topics[0], predicted_subtopics[0]

# 🔹 Run Predictions for All Samples
topic_predictions = []
subtopic_predictions = []
for text in test_texts:
    topic_pred, subtopic_pred = predict_topics_faiss(text)
    topic_predictions.append(topic_pred)
    subtopic_predictions.append(subtopic_pred)

topic_predictions = np.array(topic_predictions)
subtopic_predictions = np.array(subtopic_predictions)

# 🔹 Compute Evaluation Metrics
def evaluate(true_labels, predicted_labels, label_type="Topic"):
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average="micro", zero_division=0)
    
    try:
        auc_roc = roc_auc_score(true_labels, predicted_labels, average="micro")
    except ValueError:
        auc_roc = "⚠️ Could not be calculated"

    print(f"\n📊 **{label_type} Evaluation Metrics:**")
    print(f"🔹 Precision: {precision:.4f}")
    print(f"🔹 Recall: {recall:.4f}")
    print(f"🔹 F1 Score: {f1:.4f}")
    print(f"🔹 AUC-ROC Score: {auc_roc}")

# 🔹 Print Evaluation Results for Topics and Subtopics
evaluate(test_topic_labels, topic_predictions, "Topic")
evaluate(test_subtopic_labels, subtopic_predictions, "Subtopic")
