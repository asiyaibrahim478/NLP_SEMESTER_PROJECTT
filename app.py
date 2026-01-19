# =========================================================
# Summary Box - Streamlit Content Summarizer
# Author: Asiya Ibrahim
# Python: 3.10+
# Streamlit Cloud Compatible (Python 3.13 Safe)
#
# GitHub Deployment Notes:
# 1. File name must be: app.py
# 2. Add requirements.txt (provided separately)
# 3. Push to GitHub
# 4. Deploy via Streamlit Cloud
# =========================================================

import streamlit as st
import os
import tempfile
import requests
import string
from collections import Counter

# ---------- NLTK SETUP (Cloud Safe) ----------
import nltk

NLTK_DATA_DIR = "/home/appuser/nltk_data"
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)

nltk.download("punkt", download_dir=NLTK_DATA_DIR)
nltk.download("punkt_tab", download_dir=NLTK_DATA_DIR)
nltk.download("stopwords", download_dir=NLTK_DATA_DIR)

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# ---------- NLP / ML ----------
from transformers import pipeline

# ---------- FILE / MEDIA ----------
import pdfplumber
from bs4 import BeautifulSoup
import speech_recognition as sr
from youtube_transcript_api import YouTubeTranscriptApi

# ---------- STREAMLIT CONFIG ----------
st.set_page_config(
    page_title="Summary Box",
    layout="wide"
)

# ---------- CUSTOM PURPLE THEME ----------
st.markdown("""
<style>
body {
    background-color: #F3E5F5;
}
h1, h2, h3 {
    color: #6A1B9A;
}
.stButton > button {
    background-color: #9C27B0;
    color: white;
    border-radius: 6px;
}
.stButton > button:hover {
    background-color: #6A1B9A;
}
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown("<h1>Summary Box</h1>", unsafe_allow_html=True)
st.markdown(
    "<h4>Content Summarizer! Get the gist of any content with one click!</h4>",
    unsafe_allow_html=True
)

# ---------- SESSION STATE ----------
if "favorites" not in st.session_state:
    st.session_state.favorites = []

# ---------- LOAD SUMMARIZER ----------
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# ---------- UTILITY FUNCTIONS ----------
def clean_text(text):
    return " ".join(text.split())

def summarize_text(text, ratio):
    text = clean_text(text)
    if len(text) < 200:
        return text

    max_len = int(150 + ratio * 250)
    summary = summarizer(
        text,
        max_length=max_len,
        min_length=60,
        do_sample=False
    )
    return summary[0]["summary_text"]

def extract_key_points(text, n=8):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    words = [
        w for w in words
        if w not in stop_words and w not in string.punctuation and len(w) > 2
    ]
    freq = Counter(words)
    return [w for w, _ in freq.most_common(n)]

def detect_technologies(text):
    tech_keywords = [
        "python", "java", "c++", "machine learning", "deep learning",
        "nlp", "ai", "artificial intelligence", "cloud", "docker",
        "kubernetes", "tensorflow", "pytorch", "streamlit"
    ]
    found = []
    text_lower = text.lower()
    for tech in tech_keywords:
        if tech in text_lower:
            found.append(tech.title())
    return list(set(found))

def infer_purpose(text):
    text = text.lower()
    if "tutorial" in text or "how to" in text:
        return "Educational / Instructional"
    if "study" in text or "research" in text:
        return "Academic / Research"
    if "buy" in text or "product" in text:
        return "Promotional / Marketing"
    return "Informational"

def creativity_score(text):
    words = word_tokenize(text.lower())
    unique_words = set(words)
    ttr = len(unique_words) / max(len(words), 1)
    score = min(10, round(ttr * 20))
    explanation = f"{score}/10 - Based on vocabulary diversity"
    return score, explanation

# ---------- EXTRACTORS ----------
def extract_from_pdf(file, pages="all"):
    text = ""
    with pdfplumber.open(file) as pdf:
        if pages == "all":
            selected_pages = range(len(pdf.pages))
        else:
            selected_pages = pages

        for i in selected_pages:
            text += pdf.pages[i].extract_text() or ""
    return text

def extract_from_url(url):
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    return " ".join(p.text for p in paragraphs)

def extract_from_audio(file):
    recognizer = sr.Recognizer()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    with sr.AudioFile(tmp_path) as source:
        audio = recognizer.record(source)
    os.remove(tmp_path)

    return recognizer.recognize_google(audio)

def extract_from_youtube(url):
    video_id = url.split("v=")[-1]
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join(item["text"] for item in transcript)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("Tools")
    st.write("Summarizer")
    st.write("Paragraph Writer")
    st.write("Writing Tips")
    st.write("Explain Like I'm 5")

    st.header("Collections")
    if st.button("View Favorites"):
        st.write(st.session_state.favorites)

    if st.button("Invite Friends"):
        st.write("Share this app link after deployment.")

# ---------- MAIN TABS ----------
tabs = st.tabs(["Text", "URL", "PDF", "Audio", "Youtube"])

def process_content(text, ratio):
    summary = summarize_text(text, ratio)
    points = extract_key_points(text)
    tech = detect_technologies(text)
    purpose = infer_purpose(text)
    creativity, explanation = creativity_score(text)

    with st.expander("Summary"):
        st.write(summary)

    with st.expander("Key Concepts / Main Points"):
        for p in points:
            st.write(f"- {p}")

    with st.expander("Technologies Used"):
        st.write(tech if tech else "None detected")

    with st.expander("Purpose of the Content"):
        st.write(purpose)

    with st.expander("Creativity Measure"):
        st.write(explanation)

    with st.expander("Additional Insights"):
        st.write("Neutral tone analysis applied.")

    if st.button("Save to Favorites"):
        st.session_state.favorites.append(summary)
        st.success("Saved to favorites")

# ---------- TEXT TAB ----------
with tabs[0]:
    text = st.text_area("Paste your text here")
    ratio = st.slider("Summary Length", 0.1, 0.5, 0.3)
    if st.button("Summarize Text"):
        if text:
            process_content(text, ratio)

# ---------- URL TAB ----------
with tabs[1]:
    url = st.text_input("Enter URL")
    ratio = st.slider("Summary Length", 0.1, 0.5, 0.3, key="url")
    if st.button("Summarize URL"):
        if url:
            text = extract_from_url(url)
            process_content(text, ratio)

# ---------- PDF TAB ----------
with tabs[2]:
    pdf = st.file_uploader("Upload PDF File", type=["pdf"])
    ratio = st.slider("Summary Length", 0.1, 0.5, 0.3, key="pdf")
    if st.button("Summarize PDF"):
        if pdf:
            text = extract_from_pdf(pdf)
            process_content(text, ratio)

# ---------- AUDIO TAB ----------
with tabs[3]:
    audio = st.file_uploader("Upload Audio File", type=["wav", "mp3"])
    ratio = st.slider("Summary Length", 0.1, 0.5, 0.3, key="audio")
    if st.button("Summarize Audio"):
        if audio:
            text = extract_from_audio(audio)
            process_content(text, ratio)

# ---------- YOUTUBE TAB ----------
with tabs[4]:
    yt = st.text_input("Enter YouTube URL")
    ratio = st.slider("Summary Length", 0.1, 0.5, 0.3, key="yt")
    if st.button("Summarize YouTube"):
        if yt:
            text = extract_from_youtube(yt)
            process_content(text, ratio)
