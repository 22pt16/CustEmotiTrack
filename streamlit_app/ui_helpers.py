import json
from src.emotion_detection import detect_emotions
from src.topic_detection import extract_topics
from src.adorescore import calculate_adorescore  # Placeholder

def process_feedback(feedback_text):
    """
    Processes customer feedback using Emotion Detection and Topic Analysis.
    Returns results in the required JSON format.
    """

    # Emotion Detection
    emotions = detect_emotions(feedback_text)

    # Topic Detection
    topics = extract_topics(feedback_text)

    # Adorescore Calculation
    adorescore = calculate_adorescore(emotions, topics)  # Placeholder function

    # Format Output JSON
    output = {
        "emotions": emotions,
        "topics": topics,
        "adorescore": adorescore
    }

    return json.dumps(output, indent=4)
