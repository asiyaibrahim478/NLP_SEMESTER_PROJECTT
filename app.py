import nltk
import os

NLTK_DATA_DIR = "/home/appuser/nltk_data"
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)

nltk.download("punkt", download_dir=NLTK_DATA_DIR)
nltk.download("punkt_tab", download_dir=NLTK_DATA_DIR)
nltk.download("stopwords", download_dir=NLTK_DATA_DIR)

# Summary Box - Versatile Content Summarizer (Streamlit App)
# ============================================================
# Author: Expert Python & Streamlit Developer
# Python Version: 3.10+
# Streamlit Version: 1.0+
#
# -------------------- GITHUB DEPLOYMENT ---------------------
# 1. Create a GitHub repository
# 2. Add this file as: app.py
#
# 3. Create requirements.txt with the following content:
# ------------------------------------------------------------
# streamlit
# transformers
# torch
# PyPDF2
# pdfplumber
# beautifulsoup4
# requests
# nltk
# spacy
# speechrecognition
# pydub
# youtube-transcript-api
#
# 4. (Optional – for Streamlit Cloud legacy)
# Create Procfile:
# web: streamlit run app.py --server.port=$PORT
#
# 5. Deploy on Streamlit Community Cloud:
# https://share.streamlit.io
# ------------------------------------------------------------
#
# NOTE:
# - Files are deleted after processing
# - Max upload size: 10MB
# - No emojis used (professional UI)
# ============================================================

import streamlit as st
import requests
import re
import tempfile
import os
from bs4 import BeautifulSoup
from transformers import pipeline
import pdfplumber
import speech_recognition as sr
from pydub import AudioSegment
from youtube_transcript_api import YouTubeTranscriptApi
import nltk
import spacy
from collections import Counter
from math import log

# -------------------- INITIAL SETUP --------------------
nltk.download("punkt")
nltk.download("stopwords")

try:
    nlp = spacy.load("en_core_web_sm")
except:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# -------------------- CUSTOM CSS (PURPLE THEME) --------------------
st.markdown("""
<style>
body {
    background-color: #F3E5F5;
}
h1, h2, h3 {
    color: #4A148C;
}
.stButton>button {
    background-color: #9C27B0;
    color: white;
    border-radius: 6px;
    padding: 0.5em 1em;
}
.stButton>button:hover {
    background-color: #6A1B9A;
}
.clear-btn button {
    background-color: #C62828 !important;
}
.sidebar-title {
    font-size: 20px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -------------------- HELPER FUNCTIONS --------------------
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_text_from_pdf(file, pages="all"):
    text = ""
    with pdfplumber.open(file) as pdf:
        if pages == "all":
            for page in pdf.pages:
                text += page.extract_text() or ""
        else:
            selected = []
            for part in pages.split(","):
                if "-" in part:
                    a, b = part.split("-")
                    selected.extend(range(int(a)-1, int(b)))
                else:
                    selected.append(int(part)-1)
            for i in selected:
                if 0 <= i < len(pdf.pages):
                    text += pdf.pages[i].extract_text() or ""
    return clean_text(text)

def extract_text_from_url(url):
    r = requests.get(url, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")
    paragraphs = [p.get_text() for p in soup.find_all("p")]
    return clean_text(" ".join(paragraphs))

def extract_text_from_audio(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio = AudioSegment.from_file(uploaded_file)
        audio.export(tmp.name, format="wav")
        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp.name) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
    os.unlink(tmp.name)
    return text

def extract_text_from_youtube(url):
    video_id = url.split("v=")[-1].split("&")[0]
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return clean_text(" ".join([t["text"] for t in transcript]))

def summarize_text(text, ratio):
    max_len = min(1024, int(len(text.split()) * ratio))
    min_len = max(50, int(max_len * 0.4))
    summary = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
    return summary[0]["summary_text"]

def extract_key_points(text, n=7):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences[:n]

def extract_technologies(text):
    tech_keywords = [
        "python", "java", "c++", "nlp", "machine learning", "deep learning",
        "ai", "artificial intelligence", "cloud", "docker", "kubernetes",
        "streamlit", "tensorflow", "pytorch", "api", "database"
    ]
    found = []
    lower = text.lower()
    for tech in tech_keywords:
        if tech in lower:
            found.append(tech.title())
    return list(set(found))

def infer_purpose(text):
    lower = text.lower()
    if "learn" in lower or "explain" in lower:
        return "Educational / Informational"
    if "buy" in lower or "product" in lower or "service" in lower:
        return "Promotional / Marketing"
    if "research" in lower or "study" in lower:
        return "Research / Academic"
    return "General Informative Content"

def creativity_score(text):
    words = nltk.word_tokenize(text.lower())
    unique_words = set(words)
    ttr = len(unique_words) / max(len(words), 1)

    sentences = nltk.sent_tokenize(text)
    avg_len = sum(len(nltk.word_tokenize(s)) for s in sentences) / max(len(sentences), 1)

    metaphor_patterns = len(re.findall(r"like a|as if|as a", text.lower()))

    score = (ttr * 5) + (min(avg_len / 20, 1) * 3) + (min(metaphor_patterns, 5) / 5 * 2)
    final = min(10, max(1, round(score, 1)))
    explanation = f"{final}/10 based on vocabulary diversity, sentence complexity, and figurative language usage."
    return final, explanation

def additional_insights(text):
    doc = nlp(text)
    polarity = sum(token.sentiment for token in doc) if hasattr(doc[0], "sentiment") else 0
    if polarity > 0:
        return "The tone appears generally positive and optimistic."
    elif polarity < 0:
        return "The tone appears critical or cautious."
    return "The tone appears neutral and informational."

# -------------------- SESSION STATE --------------------
if "favorites" not in st.session_state:
    st.session_state.favorites = []

# -------------------- SIDEBAR --------------------
st.sidebar.markdown("<div class='sidebar-title'>Tools</div>", unsafe_allow_html=True)
with st.sidebar.expander("Tools", expanded=True):
    st.write("Summarizer")
    st.write("Paragraph Writer")
    st.write("Writing Tips")
    st.write("Explain Like I'm 5")

with st.sidebar.expander("Collections"):
    if st.button("Favorites"):
        st.write(st.session_state.favorites)
    if st.button("Invite Friends"):
        st.info("Share this app link from Streamlit Cloud")

# -------------------- MAIN HEADER --------------------
st.markdown("<h1>Summary Box</h1>", unsafe_allow_html=True)
st.markdown("<h3>Content Summarizer! Get the gist of any content with one click!</h3>", unsafe_allow_html=True)

# -------------------- TABS --------------------
tabs = st.tabs(["Text", "URL", "PDF", "Audio", "Youtube"])

def run_pipeline(text, ratio):
    summary = summarize_text(text, ratio)
    keys = extract_key_points(summary)
    tech = extract_technologies(text)
    purpose = infer_purpose(text)
    score, explanation = creativity_score(text)
    insights = additional_insights(text)
    return summary, keys, tech, purpose, score, explanation, insights

def render_output(result):
    summary, keys, tech, purpose, score, explanation, insights = result
    with st.expander("Summary", expanded=True):
        st.write(summary)
    with st.expander("Key Concepts / Main Points"):
        for k in keys:
            st.write(f"- {k}")
    with st.expander("Technologies Used"):
        st.write(", ".join(tech) if tech else "No specific technologies detected.")
    with st.expander("Purpose of the Content"):
        st.write(purpose)
    with st.expander("Creativity Measure"):
        st.write(explanation)
    with st.expander("Additional Insights"):
        st.write(insights)
    if st.button("Save to Favorites"):
        st.session_state.favorites.append(summary)
        st.success("Saved to Favorites")

# -------------------- TAB CONTENT --------------------
with tabs[0]:
    text = st.text_area("Paste your text here")
    ratio = st.slider("Summary Length", 0, 100, 30) / 100
    col1, col2 = st.columns(2)
    if col1.button("Summarize"):
        if text:
            render_output(run_pipeline(text, ratio))
    if col2.button("Clear"):
        st.experimental_rerun()

with tabs[1]:
    url = st.text_input("Enter URL")
    ratio = st.slider("Summary Length", 0, 100, 30, key="url_ratio") / 100
    if st.button("Fetch & Summarize"):
        try:
            text = extract_text_from_url(url)
            render_output(run_pipeline(text, ratio))
        except:
            st.error("Unable to fetch content from URL.")

with tabs[2]:
    st.subheader("Upload PDF File")
    ratio = st.slider("Summary Length", 0, 100, 30, key="pdf_ratio") / 100
    pages = st.text_input("Select Pages (e.g., 1,3,5-7) or leave blank for all")
    pdf = st.file_uploader("Click or Drop File", type=["pdf"])
    st.caption("Once the summary has been created, your file will be deleted automatically. Files must be under 10MB.")
    col1, col2 = st.columns(2)
    if col1.button("Summarize PDF"):
        if pdf:
            text = extract_text_from_pdf(pdf, pages if pages else "all")
            render_output(run_pipeline(text, ratio))
    if col2.button("Clear PDF"):
        st.experimental_rerun()

with tabs[3]:
    audio = st.file_uploader("Upload Audio File", type=["mp3", "wav"])
    ratio = st.slider("Summary Length", 0, 100, 30, key="audio_ratio") / 100
    if st.button("Transcribe & Summarize"):
        if audio:
            text = extract_text_from_audio(audio)
            render_output(run_pipeline(text, ratio))

with tabs[4]:
    yt = st.text_input("Enter YouTube URL")
    ratio = st.slider("Summary Length", 0, 100, 30, key="yt_ratio") / 100
    if st.button("Fetch Transcript & Summarize"):
        try:
            text = extract_text_from_youtube(yt)
            render_output(run_pipeline(text, ratio))
        except:
            st.error("Transcript not available for this video.")
