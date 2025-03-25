import json
import torch
import pickle
import os
from sentence_transformers import SentenceTransformer

# 🔹 Initialize Model
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# 🔹 Load Topics & Subtopics from a Local JSON File
topic_mapping_path = "datasets/topic_mapping.json"  # Adjust path if needed
with open(topic_mapping_path, "r", encoding="utf-8") as file:
    topic_subtopic_mapping = json.load(file)

# 🔹 Encode Topics
topics = list(topic_subtopic_mapping.keys())
topic_embeddings = embedder.encode(topics, convert_to_tensor=True)

# 🔹 Save Model & Embeddings Locally
model_data = {
    "model_name": "all-MiniLM-L6-v2",
    "topic_mapping": topic_subtopic_mapping,
    "topic_embeddings": topic_embeddings.cpu().numpy(),
    "topics": topics
}

# 🔹 Create Directory if Not Exists
save_dir = "models"
os.makedirs(save_dir, exist_ok=True)

# 🔹 Save to Local File
save_path = os.path.join(save_dir, "topic_model.pkl")
with open(save_path, "wb") as f:
    pickle.dump(model_data, f)

print(f"✅ Topic model saved locally at {save_path}")
