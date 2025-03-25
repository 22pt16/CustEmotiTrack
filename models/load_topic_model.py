import json
import torch
import pickle
import os
from sentence_transformers import SentenceTransformer, util

# 🔹 Load the Saved Model from "models/topic_model.pkl"
model_path = "models/topic_model.pkl"
with open(model_path, "rb") as f:
    model_data = pickle.load(f)

# 🔹 Load Sentence Transformer Model
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer(model_data["model_name"], device=device)

# 🔹 Extract Topics & Encodings from Loaded Model
topic_subtopic_mapping = model_data["topic_mapping"]
topics = model_data["topics"]
topic_embeddings = torch.tensor(model_data["topic_embeddings"], device=device)  # Load embeddings on same device

def predict_topics(review, top_k_topics=3, top_k_subtopics=2):
    """Predicts Top-3 Main Topics and Top-2 Subtopics using sentence embeddings."""

    # 🔹 Encode Input Review
    review_embedding = embedder.encode(review, convert_to_tensor=True)

    # 🔹 Find Top-K Topics (Cosine Similarity)
    topic_scores = util.pytorch_cos_sim(review_embedding, topic_embeddings)[0]
    top_topic_indices = torch.topk(topic_scores, top_k_topics).indices.tolist()

    main_topics = [
        {"topic": topics[i], "confidence": round(float(topic_scores[i]), 2)}
        for i in top_topic_indices
    ]

    # 🔹 Find Top-K Subtopics for Each Main Topic
    subtopics_result = {}
    for main_topic in main_topics:
        topic_name = main_topic["topic"]  # ✅ Fixed: Using string key
        subtopic_groups = topic_subtopic_mapping.get(topic_name, {}).get("subtopics", {})

        # Flatten subtopics list
        subtopics = [sub for group in subtopic_groups.values() for sub in group]

        if subtopics:
            subtopic_embeddings = embedder.encode(subtopics, convert_to_tensor=True)
            subtopic_scores = util.pytorch_cos_sim(review_embedding, subtopic_embeddings)[0]
            top_subtopic_values, top_subtopic_indices = torch.topk(subtopic_scores, min(top_k_subtopics, len(subtopics)))

            # Store subtopics with confidence scores
            subtopics_result[topic_name] = [
                {"subtopic": subtopics[i], "confidence": round(float(top_subtopic_values[j]), 2)}
                for j, i in enumerate(top_subtopic_indices.tolist())
            ]
        else:
            subtopics_result[topic_name] = []

    return {
        "topics": {
            "main": main_topics,
            "subtopics": subtopics_result
        }
    }
'''
# 🔹 Example Reviews
reviews = [
    "The delivery was very late, and there were no tracking updates.",
    "The product material is of very poor quality and feels cheap.",
    "Very bad design"
]

# 🔹 Predict Topics & Subtopics for Each Review
results = [predict_topics(review) for review in reviews]

# 🔹 Print Results
print(json.dumps(results, indent=4))
'''