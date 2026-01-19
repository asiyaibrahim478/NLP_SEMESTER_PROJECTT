import streamlit as st
import os
import tempfile
import re
from pathlib import Path
import nltk
from collections import Counter

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# File processing imports
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from docx import Document
except ImportError:
    Document = None

# NLP and ML imports
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Configuration
st.set_page_config(
    page_title="Advanced Text Summarizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

@st.cache_data
def load_models():
    """Load NLP models with caching"""
    models = {}
    if TRANSFORMERS_AVAILABLE:
        try:
            models['summarizer'] = pipeline("summarization", model="facebook/bart-large-cnn")
            models['classifier'] = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
            models['sentiment'] = pipeline("sentiment-analysis")
        except Exception as e:
            st.warning(f"Could not load transformer models: {e}")
    return models

def extract_text_from_pdf(file):
    """Extract text from PDF files"""
    if not PyPDF2:
        return None, "PyPDF2 not installed. Please install it to process PDF files."
    
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = []
        for page in pdf_reader.pages:
            text.append(page.extract_text())
        return "\n".join(text), None
    except Exception as e:
        return None, f"Error reading PDF: {str(e)}"

def extract_text_from_docx(file):
    """Extract text from DOCX files"""
    if not Document:
        return None, "python-docx not installed. Please install it to process DOCX files."
    
    try:
        doc = Document(file)
        text = []
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
        return "\n".join(text), None
    except Exception as e:
        return None, f"Error reading DOCX: {str(e)}"

def extract_text_from_txt(file):
    """Extract text from TXT/MD files"""
    try:
        return file.read().decode('utf-8'), None
    except UnicodeDecodeError:
        try:
            file.seek(0)
            return file.read().decode('latin-1'), None
        except Exception as e:
            return None, f"Error reading text file: {str(e)}"

def preprocess_text(text):
    """Clean and normalize text"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove page numbers (common patterns)
    text = re.sub(r'\b\d+\b(?=\s*$)', '', text, flags=re.MULTILINE)
    return text.strip()

def extract_file_content(uploaded_file):
    """Main function to extract text from uploaded files"""
    file_extension = Path(uploaded_file.name).suffix.lower()
    
    if file_extension == '.pdf':
        text, error = extract_text_from_pdf(uploaded_file)
    elif file_extension in ['.docx', '.doc']:
        text, error = extract_text_from_docx(uploaded_file)
    elif file_extension in ['.txt', '.md', '.markdown']:
        text, error = extract_text_from_txt(uploaded_file)
    else:
        return None, f"Unsupported file type: {file_extension}"
    
    if error:
        return None, error
    
    if not text or len(text.strip()) < 50:
        return None, "File appears to be empty or contains insufficient text."
    
    return preprocess_text(text), None

def generate_extractive_summary(text, num_sentences=5):
    """Generate extractive summary using sentence scoring"""
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text
    
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [w for w in words if w.isalnum() and w not in stop_words]
    
    # Calculate word frequencies
    word_freq = Counter(words)
    
    # Score sentences
    sentence_scores = {}
    for sentence in sentences:
        sentence_words = word_tokenize(sentence.lower())
        sentence_words = [w for w in sentence_words if w.isalnum()]
        score = sum(word_freq.get(w, 0) for w in sentence_words)
        sentence_scores[sentence] = score / (len(sentence_words) + 1)
    
    # Get top sentences
    top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
    top_sentences = sorted(top_sentences, key=lambda x: sentences.index(x[0]))
    
    return " ".join([s[0] for s in top_sentences])

def generate_summary(text, length="medium", models=None):
    """Generate text summary using transformer or extractive methods"""
    if TRANSFORMERS_AVAILABLE and models and 'summarizer' in models:
        try:
            # Determine max length based on user preference
            length_map = {
                "short": (100, 150),
                "medium": (200, 300),
                "long": (300, 500)
            }
            min_len, max_len = length_map.get(length, (200, 300))
            
            # Chunk text if too long (BART max is ~1024 tokens)
            max_input = 1024
            words = text.split()
            if len(words) > max_input:
                text = " ".join(words[:max_input])
            
            summary = models['summarizer'](text, max_length=max_len, min_length=min_len, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            st.warning(f"Transformer summarization failed, using extractive method: {e}")
    
    # Fallback to extractive summarization
    num_sentences_map = {"short": 3, "medium": 5, "long": 8}
    num_sentences = num_sentences_map.get(length, 5)
    return generate_extractive_summary(text, num_sentences)

def extract_key_concepts(text, num_concepts=10):
    """Extract key concepts using frequency analysis and filtering"""
    stop_words = set(stopwords.words('english'))
    
    # Tokenize and filter
    words = word_tokenize(text.lower())
    words = [w for w in words if w.isalnum() and len(w) > 3 and w not in stop_words]
    
    # Get most common words
    word_freq = Counter(words)
    common_words = word_freq.most_common(num_concepts * 2)
    
    # Extract noun phrases (simple approach)
    sentences = sent_tokenize(text)
    phrases = []
    for sentence in sentences[:50]:  # Limit to first 50 sentences for performance
        # Look for capitalized phrases (potential named entities)
        caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sentence)
        phrases.extend(caps)
    
    phrase_freq = Counter(phrases)
    
    # Combine words and phrases
    concepts = []
    for phrase, count in phrase_freq.most_common(5):
        if count > 1:
            concepts.append(phrase)
    
    for word, count in common_words:
        if len(concepts) >= num_concepts:
            break
        if word not in [c.lower() for c in concepts]:
            concepts.append(word.title())
    
    return concepts[:num_concepts]

def identify_technologies(text):
    """Identify technologies and tools mentioned in the text"""
    # Common tech keywords and patterns
    tech_patterns = {
        'Languages': r'\b(Python|Java|JavaScript|C\+\+|Ruby|PHP|Swift|Kotlin|Go|Rust|TypeScript|R|MATLAB|Scala|Perl)\b',
        'Frameworks': r'\b(React|Angular|Vue|Django|Flask|Spring|Express|Laravel|Rails|TensorFlow|PyTorch|Keras|scikit-learn)\b',
        'Databases': r'\b(MySQL|PostgreSQL|MongoDB|Redis|Oracle|SQL Server|SQLite|Cassandra|DynamoDB|Firebase)\b',
        'Cloud/DevOps': r'\b(AWS|Azure|Google Cloud|GCP|Docker|Kubernetes|Jenkins|GitLab|CircleCI|Terraform|Ansible)\b',
        'Tools': r'\b(Git|GitHub|VSCode|Jupyter|Streamlit|Apache|Nginx|ElasticSearch|Kafka|RabbitMQ)\b',
    }
    
    found_technologies = {}
    for category, pattern in tech_patterns.items():
        matches = re.finditer(pattern, text, re.IGNORECASE)
        techs = set()
        for match in matches:
            techs.add(match.group())
        if techs:
            found_technologies[category] = list(techs)
    
    return found_technologies

def infer_document_purpose(text, models=None):
    """Infer the purpose of the document"""
    candidate_labels = [
        "technical documentation",
        "research paper",
        "business proposal",
        "educational material",
        "code documentation",
        "user manual",
        "report",
        "tutorial"
    ]
    
    if TRANSFORMERS_AVAILABLE and models and 'classifier' in models:
        try:
            # Use first 512 words for classification
            sample = " ".join(text.split()[:512])
            result = models['classifier'](sample, candidate_labels)
            top_label = result['labels'][0]
            confidence = result['scores'][0]
            return f"This document appears to be **{top_label}** (confidence: {confidence:.2%}). "
        except Exception as e:
            st.warning(f"Classification failed: {e}")
    
    # Fallback: rule-based inference
    text_lower = text.lower()
    
    if any(word in text_lower for word in ['abstract', 'methodology', 'conclusion', 'references']):
        return "This document appears to be a **research paper** based on its structure. "
    elif any(word in text_lower for word in ['install', 'usage', 'import', 'function', 'class']):
        return "This document appears to be **technical documentation** or a **tutorial**. "
    elif any(word in text_lower for word in ['proposal', 'executive summary', 'budget', 'timeline']):
        return "This document appears to be a **business proposal** or **report**. "
    else:
        return "This document appears to be **informational content**. "

def analyze_sentiment(text, models=None):
    """Analyze overall sentiment of the text"""
    if TRANSFORMERS_AVAILABLE and models and 'sentiment' in models:
        try:
            # Use first 512 words
            sample = " ".join(text.split()[:512])
            result = models['sentiment'](sample)
            label = result[0]['label']
            score = result[0]['score']
            return f"{label.title()} (confidence: {score:.2%})"
        except Exception as e:
            return f"Unable to determine (error: {e})"
    
    # Simple fallback
    positive_words = ['good', 'great', 'excellent', 'positive', 'success', 'benefit']
    negative_words = ['bad', 'poor', 'negative', 'failure', 'problem', 'issue']
    
    text_lower = text.lower()
    pos_count = sum(text_lower.count(word) for word in positive_words)
    neg_count = sum(text_lower.count(word) for word in negative_words)
    
    if pos_count > neg_count * 1.5:
        return "Positive"
    elif neg_count > pos_count * 1.5:
        return "Negative"
    else:
        return "Neutral"

def generate_additional_insights(text, models=None):
    """Generate additional insights about the document"""
    insights = {}
    
    # Sentiment analysis
    insights['Sentiment'] = analyze_sentiment(text, models)
    
    # Word count statistics
    words = text.split()
    insights['Word Count'] = len(words)
    insights['Sentence Count'] = len(sent_tokenize(text))
    insights['Avg Words per Sentence'] = f"{len(words) / max(len(sent_tokenize(text)), 1):.1f}"
    
    # Reading time estimate (average 200 words per minute)
    insights['Estimated Reading Time'] = f"{len(words) / 200:.1f} minutes"
    
    # Detect questions
    questions = re.findall(r'[^.!?]*\?', text)
    if questions:
        insights['Questions Found'] = len(questions)
    
    return insights

def format_output_text(result):
    """Format analysis result as downloadable text"""
    output = []
    output.append("=" * 60)
    output.append("ADVANCED TEXT ANALYSIS REPORT")
    output.append("=" * 60)
    output.append(f"\nFile: {result['filename']}")
    output.append(f"Type: {result['file_type']}")
    output.append(f"Word Count: {result['word_count']}")
    output.append("\n" + "=" * 60)
    
    output.append("\n\nSUMMARY")
    output.append("-" * 60)
    output.append(result['summary'])
    
    output.append("\n\nKEY CONCEPTS")
    output.append("-" * 60)
    for i, concept in enumerate(result['key_concepts'], 1):
        output.append(f"{i}. {concept}")
    
    if result['technologies']:
        output.append("\n\nTECHNOLOGIES IDENTIFIED")
        output.append("-" * 60)
        for category, techs in result['technologies'].items():
            output.append(f"\n{category}:")
            for tech in techs:
                output.append(f"  - {tech}")
    
    output.append("\n\nDOCUMENT PURPOSE")
    output.append("-" * 60)
    output.append(result['purpose'])
    
    output.append("\n\nADDITIONAL INSIGHTS")
    output.append("-" * 60)
    for key, value in result['insights'].items():
        output.append(f"{key}: {value}")
    
    output.append("\n" + "=" * 60)
    output.append("End of Report")
    output.append("=" * 60)
    
    return "\n".join(output)

def main():
    # Header
    st.markdown('<p class="main-header">📄 Advanced Text Summarizer</p>', unsafe_allow_html=True)
    st.markdown("Upload a document and get comprehensive analysis including summary, key concepts, technologies, and more!")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    summary_length = st.sidebar.select_slider(
        "Summary Length",
        options=["short", "medium", "long"],
        value="medium"
    )
    
    st.sidebar.subheader("Include Sections")
    include_summary = st.sidebar.checkbox("Summary", value=True)
    include_concepts = st.sidebar.checkbox("Key Concepts", value=True)
    include_tech = st.sidebar.checkbox("Technologies", value=True)
    include_purpose = st.sidebar.checkbox("Document Purpose", value=True)
    include_insights = st.sidebar.checkbox("Additional Insights", value=True)
    
    num_concepts = st.sidebar.slider("Number of Key Concepts", 5, 20, 10)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("This app uses advanced NLP to analyze your documents and extract meaningful insights.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'docx', 'txt', 'md', 'doc'],
        help="Supported formats: PDF, DOCX, TXT, Markdown"
    )
    
    if uploaded_file is not None:
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Name", uploaded_file.name)
        with col2:
            st.metric("File Size", f"{uploaded_file.size / 1024:.2f} KB")
        with col3:
            st.metric("File Type", Path(uploaded_file.name).suffix.upper())
        
        # Process button
        if st.button("🚀 Analyze Document", type="primary"):
            with st.spinner("Extracting text from document..."):
                text, error = extract_file_content(uploaded_file)
            
            if error:
                st.error(f"❌ {error}")
                return
            
            if not text:
                st.error("❌ Could not extract text from the document.")
                return
            
            # Load models
            with st.spinner("Loading AI models..."):
                models = load_models()
            
            # Perform analysis
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            result = {
                'filename': uploaded_file.name,
                'file_type': Path(uploaded_file.name).suffix,
                'word_count': len(text.split())
            }
            
            if include_summary:
                status_text.text("Generating summary...")
                progress_bar.progress(20)
                result['summary'] = generate_summary(text, summary_length, models)
            
            if include_concepts:
                status_text.text("Extracting key concepts...")
                progress_bar.progress(40)
                result['key_concepts'] = extract_key_concepts(text, num_concepts)
            
            if include_tech:
                status_text.text("Identifying technologies...")
                progress_bar.progress(60)
                result['technologies'] = identify_technologies(text)
            
            if include_purpose:
                status_text.text("Inferring document purpose...")
                progress_bar.progress(80)
                result['purpose'] = infer_document_purpose(text, models)
            
            if include_insights:
                status_text.text("Generating additional insights...")
                progress_bar.progress(90)
                result['insights'] = generate_additional_insights(text, models)
            
            progress_bar.progress(100)
            status_text.text("Analysis complete! ✨")
            
            st.session_state.analysis_result = result
        
        # Display results
        if st.session_state.analysis_result:
            result = st.session_state.analysis_result
            
            st.success("✅ Analysis completed successfully!")
            
            # Summary
            if 'summary' in result:
                with st.expander("📝 Summary", expanded=True):
                    st.write(result['summary'])
            
            # Key Concepts
            if 'key_concepts' in result:
                with st.expander("💡 Key Concepts", expanded=True):
                    for i, concept in enumerate(result['key_concepts'], 1):
                        st.write(f"{i}. **{concept}**")
            
            # Technologies
            if 'technologies' in result and result['technologies']:
                with st.expander("🔧 Technologies Identified", expanded=True):
                    for category, techs in result['technologies'].items():
                        st.markdown(f"**{category}:**")
                        st.write(", ".join(techs))
            
            # Purpose
            if 'purpose' in result:
                with st.expander("🎯 Document Purpose", expanded=True):
                    st.write(result['purpose'])
            
            # Additional Insights
            if 'insights' in result:
                with st.expander("📊 Additional Insights", expanded=True):
                    cols = st.columns(2)
                    items = list(result['insights'].items())
                    for i, (key, value) in enumerate(items):
                        with cols[i % 2]:
                            st.metric(key, value)
            
            # Download button
            output_text = format_output_text(result)
            st.download_button(
                label="📥 Download Analysis Report",
                data=output_text,
                file_name=f"analysis_{Path(result['filename']).stem}.txt",
                mime="text/plain"
            )
            
            # Feedback section
            st.markdown("---")
            st.subheader("📢 Feedback")
            col1, col2 = st.columns([3, 1])
            with col1:
                feedback = st.text_area("How was the analysis? Any suggestions?", height=100)
            with col2:
                rating = st.select_slider("Rating", options=[1, 2, 3, 4, 5], value=5)
            
            if st.button("Submit Feedback"):
                if feedback:
                    st.success("Thank you for your feedback! 🙏")
                else:
                    st.warning("Please enter some feedback before submitting.")

if __name__ == "__main__":
    main()
