import streamlit as st
import json
import sys
import os
import pandas as pd
import plotly.express as px
from deep_translator import GoogleTranslator

# ✅ Get the absolute path of the project's root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# ✅ Ensure 'models' directory is in the Python path
sys.path.append(MODELS_DIR)

# ✅ Import emotion and topic prediction functions
try:
    from load_model import predict_emotions
    from load_topic_model import predict_topics
except ModuleNotFoundError:
    st.error("⚠️ Error: Could not find `load_model.py` or `load_topic_model.py` in `models/`. Check the file paths.")
    st.stop()

def translate_to_english(text):
    return GoogleTranslator(source='auto', target='en').translate(text)

# ✅ Streamlit App Title
st.set_page_config(layout="wide", page_title="Customer Emotion Analysis Dashboard")
st.title("📊 Customer Emotion Analysis Dashboard")

# ✅ Sidebar for Input
with st.sidebar:
    st.header("📝 Enter Customer Feedback")
    user_input = st.text_area("Type or paste feedback here (separate by double newline):")
    # ✅ Limit k values between 1 and 3
    st.header("Choose k for Topic Prediction ")
    top_k1 = st.number_input("Top k main topics", min_value=1, max_value=3, value=1, step=1)
    top_k2 = st.number_input("Top k subtopics", min_value=1, max_value=3, value=1, step=1)
    analyze_button = st.button("Analyze Feedback")

# ✅ Session Storage for Analyzed Data
if "feedback_data" not in st.session_state:
    st.session_state.feedback_data = []

# ✅ Process Input on Button Click
if analyze_button and user_input:
    translated = translate_to_english(user_input) #Supports multilingual
    feedbacks = translated.strip().split("\n\n")  # Split by double newline

    for i, feedback in enumerate(feedbacks, start=len(st.session_state.feedback_data) + 1):
        emotion_prediction = predict_emotions(feedback)
        topic_prediction = predict_topics(feedback,top_k1, top_k2)


        # Extract Emotion Intensity
        emotion_intensity = emotion_prediction["emotions"]["primary"]["intensity"]  # (0-1 scale)
        main_topics = topic_prediction["topics"]["main"]
        subtopics_dict = topic_prediction["topics"]["subtopics"]
        emotions = emotion_prediction["emotions"]["primary"]["emotion"]

         # Map emotions to topics and subtopics
        topic_emotion_map = {topic["topic"]: emotions for topic in main_topics}

       # Weighted  Dynamically Compute Sentiment Weight
       # ✅ Extract Primary & Secondary Emotions
        primary_emotion = emotion_prediction["emotions"]["primary"]
        secondary_emotion = emotion_prediction["emotions"]["secondary"]

        # ✅ Extract Intensity and Count of Emotions
        primary_intensity = primary_emotion.get("intensity", 0.5)  # Default to 0.5 if missing
        secondary_intensity = secondary_emotion.get("intensity", 0.3)  # Default lower value
        num_emotions = len(emotion_prediction["emotions"])  # Total emotions counted

        # ✅ Scale intensities from 0-1 to (-100 to 100)
        scaled_primary_intensity = (primary_intensity * 2 - 1) * 100
        scaled_secondary_intensity = (secondary_intensity * 2 - 1) * 100

       # ✅ Extract sentiment label and score
        sentiment_label = emotion_prediction.get("sentiment", {}).get("label", "neutral").lower()
        sentiment_score = emotion_prediction.get("sentiment", {}).get("score", 1.0)

        # ✅ Assign weight based on sentiment
        sentiment_weight = {"positive": 1.5, "neutral": 1.0, "negative": 0.5}.get(sentiment_label, 1.0)

        # ✅ Compute final weighted sentiment score
        weighted_score = sentiment_score * sentiment_weight  # 🔹 Fixed NameError issue

        # ✅ Compute Adorescore Breakdown
        adorescore_breakdown = {
            topic_dict.get("topic", "unknown"): round(
                ((scaled_primary_intensity * num_emotions) + (scaled_secondary_intensity * num_emotions)) * weighted_score
            )
            for topic_dict in main_topics
        }

        # ✅ Compute Overall Adorescore
       
        raw_adorescore = sum(adorescore_breakdown.values()) / max(len(adorescore_breakdown), 1)

        # ✅ Normalize to (-100, 100) range using Min-Max Scaling
        min_score, max_score = -50, 150  # Adjust based on realistic observed values
        scaled_adorescore = ((raw_adorescore - min_score) / (max_score - min_score)) * 200 - 100
        overall_adorescore = max(-100, min(100, round(scaled_adorescore)))



        # ✅ Assign to session data
        adorescore = {
            "overall": overall_adorescore,
            "breakdown": adorescore_breakdown
        }



        # ✅ Structure Prediction with Adorescore Breakdown
        prediction = {
            "Feedback ID": i,
            "Text": feedback,
            **emotion_prediction,
            **topic_prediction,
            "adorescore": adorescore,
            "theme_emotion_map": topic_emotion_map,

            }
        
        
        st.session_state.feedback_data.append(prediction)


# ✅ Convert Session Data to DataFrame
if st.session_state.feedback_data:
    df = pd.DataFrame(st.session_state.feedback_data)
    
   # Extract emotions and activation levels
    emotions = df["emotions"].apply(lambda x: x["primary"]["emotion"])
    activations = df["emotions"].apply(lambda x: x["primary"]["activation"])
    df["Primary Emotion"] = emotions
    df["Activation Level"] = activations
    
    # ✅ Calculate KPI Metrics
    overall_adorescore = df["adorescore"].apply(lambda x: x["overall"] if isinstance(x, dict) and isinstance(x["overall"], (int, float)) else 0).mean()
    dominant_emotion = df["Primary Emotion"].value_counts().idxmax()
    dominant_percentage = round((df["Primary Emotion"].value_counts().max() / len(df)) * 100, 2)
    
    # ✅ Extract Top Themes
    all_topics = [topic["topic"] for entry in st.session_state.feedback_data for topic in entry["topics"]["main"]]
    top_topics = pd.Series(all_topics).value_counts().head(3) if all_topics else pd.Series()

    
    # ✅ Dashboard Layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📌 Key Metrics")
        st.metric("Overall Adorescore", round(overall_adorescore, 2))
        st.write(f"🛠 **Driven by** {dominant_emotion} ({dominant_percentage}%)")
        
        
        st.subheader("🏷️ Top Themes")
        if not top_topics.empty:
            top_topics_df = top_topics.reset_index()[["index"]].rename(columns={"index": "Top Themes"})
            top_themes_list = top_topics_df["Top Themes"].head(3).tolist()  # Only top 3
            formatted_themes = "\n".join([f"{i+1}. {theme}" for i, theme in enumerate(top_themes_list)])
            st.write(formatted_themes)  


    with col2:
        st.subheader("🎭 Emotion Activation Breakdown")
        activation_counts = df["Activation Level"].value_counts()
        if not activation_counts.empty:
            # ✅ Donut Chart for Activation Level Distribution
            fig_activation = px.pie(
                values=activation_counts, 
                names=activation_counts.index, 
                title="Activation Level Distribution", 
                hole=0.4  # 🔹 Creates a donut chart
            )
            st.plotly_chart(fig_activation)
        
    
        st.subheader("🎭 Theme-Emotion Correlation")
        emotion_theme_counts = {}

        for entry in st.session_state.feedback_data:
            for topic in entry["topics"]["main"]:  # ✅ Only main topics
                topic_name = topic["topic"]  # Extract topic name

                if topic_name not in emotion_theme_counts:
                    emotion_theme_counts[topic_name] = 0

                emotion_theme_counts[topic_name] += 1  # ✅ Count total occurrences

        # ✅ Convert to DataFrame & Display
        theme_df = pd.DataFrame(list(emotion_theme_counts.items()), columns=["Main Topic", "Total Emotion Count"])
        st.dataframe(theme_df)


    
    # ✅ Identify Highest Activated Emotion
    st.subheader("🚀 Most Activated Emotion")
    max_activation = df.groupby("Primary Emotion")["Activation Level"].count().idxmax()
    st.write(f"🔥 The most frequently activated emotion is **{max_activation}**")
    
    st.subheader("📝 Individual Feedback Analysis")
    for feedback in st.session_state.feedback_data:
        with st.expander(f"Feedback #{feedback['Feedback ID']}"):
            st.json(feedback)



