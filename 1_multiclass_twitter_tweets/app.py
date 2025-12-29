import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Sentiment AI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #ffffff;
    }

    /* Glassmorphism Card Effect */
    [data-testid="stVerticalBlock"] > div:has(div.stTextArea) {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        padding: 20px !important;
    }

    /* Input Area Styling */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
    }

    /* Button Styling */
    .stButton button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%) !important;
        color: white !important;
        border: none !important;
        padding: 10px 24px !important;
        border-radius: 30px !important;
        font-weight: 600 !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        width: 100% !important;
    }

    .stButton button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.4) !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.3) !important;
    }

    /* Title Styling */
    .main-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        background: -webkit-linear-gradient(#eee, #333);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 30px;
    }

</style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_classifier():
    return pipeline('text-classification', model='bert-base-uncased-sentiment-model', top_k=None)

classifier = load_classifier()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center; font-size: 5rem; margin-bottom: 0;'>🤖</h1>", unsafe_allow_html=True)
    st.title("Sentiment AI")
    st.markdown("---")
    st.markdown("### About")
    st.write("This app uses a fine-tuned BERT model to classify the sentiment of tweets into 6 categories.")
    st.markdown("### Categories")
    st.write("😢 Sadness | 😊 Joy | ❤️ Love | 💢 Anger | 😨 Fear | 😲 Surprise")

# --- MAIN PAGE ---
cols = st.columns([1, 2, 1])

with cols[1]:
    st.markdown("<h1 class='main-title' style='text-align: center;'>Sentiment Analysis</h1>", unsafe_allow_html=True)
    st.write("Experience the power of BERT-based classification.")
    
    with st.container():
        text = st.text_area("What's on your mind?", placeholder="Type your tweet here...", height=150)

        if st.button("Analyze Sentiment"):
            if text.strip() == "":
                st.warning("Please enter some text first!")
            else:
                with st.spinner("Analyzing emotions..."):
                    results = classifier(text)[0]
                    
                    st.markdown("### Results")
                    
                    # Top prediction highlight
                    top_prediction = results[0]
                    label_map = {
                        "sadness": "😢 Sadness",
                        "joy": "😊 Joy",
                        "love": "❤️ Love",
                        "anger": "💢 Anger",
                        "fear": "😨 Fear",
                        "surprise": "😲 Surprise"
                    }
                    
                    display_label = label_map.get(top_prediction['label'], top_prediction['label'])
                    
                    st.success(f"Primary Emotion: **{display_label}**")
                    
                    st.markdown("---")
                    
                    # Detailed breakdown
                    for result in results:
                        label = result['label']
                        score = result['score']
                        emoji_label = label_map.get(label, label)
                        
                        col_l, col_r = st.columns([1, 4])
                        col_l.write(emoji_label)
                        col_r.progress(score)
                        
        st.markdown("---")
        st.caption("Powered by Hugging Face Transformers & Streamlit")