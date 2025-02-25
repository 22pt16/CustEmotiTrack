import streamlit as st
import json
from ui_helpers import process_feedback  # Function to call emotion & topic modules

# Streamlit UI Setup
st.set_page_config(page_title="Customer Emotion Analysis", layout="wide")

# App Title
st.title("📝 Customer Emotion Analysis System")
st.markdown("Analyze customer feedback to extract emotions, topics, and Adorescore.")

# User Input Section
feedback_text = st.text_area("Enter Customer Feedback:", height=150, placeholder="Type customer feedback here...")

# Process Feedback Button
if st.button("Analyze Feedback"):
    if feedback_text.strip():
        # Call processing function
        output_data = process_feedback(feedback_text)

        # Display JSON output
        st.subheader("📊 Analysis Result:")
        st.json(output_data, expanded=True)
    else:
        st.warning("Please enter customer feedback.")

# Footer
st.markdown("---")
st.caption("Developed for Customer Sentiment Analysis.")

