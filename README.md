# Advanced Text Summarization App

A powerful text analysis and summarization tool built with Streamlit that provides comprehensive insights from your documents using advanced NLP and machine learning techniques.

## Features

- **Multi-Format Support**: Upload PDF, DOCX, TXT, and Markdown files
- **AI-Powered Summarization**: Generate concise summaries using transformer models (BART)
- **Key Concept Extraction**: Automatically identify main topics and important terms
- **Technology Detection**: Find mentions of programming languages, frameworks, and tools
- **Document Classification**: Infer the purpose and type of document
- **Sentiment Analysis**: Understand the overall tone of the content
- **Statistical Insights**: Get word counts, reading time, and more
- **Customizable Output**: Control summary length and included sections
- **Export Results**: Download analysis reports as text files

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Text-Summarization-App.git
   cd Text-Summarization-App
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (automatic on first run, but you can pre-download)
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

## Usage

### Running Locally

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to `http://localhost:8501`

3. Upload a document and configure your analysis preferences in the sidebar

4. Click "Analyze Document" to process your file

5. Review the results and download the analysis report

### Deploying to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select `app.py` as the main file
5. Deploy!

## Configuration Options

### Sidebar Settings

- **Summary Length**: Choose between short (3 sentences), medium (5 sentences), or long (8 sentences)
- **Include Sections**: Toggle which analysis sections to display
  - Summary
  - Key Concepts
  - Technologies
  - Document Purpose
  - Additional Insights
- **Number of Key Concepts**: Adjust how many key concepts to extract (5-20)

## Supported File Formats

- **PDF** (.pdf): Extracted using PyPDF2
- **Word Documents** (.docx, .doc): Parsed with python-docx
- **Text Files** (.txt): Direct text reading
- **Markdown** (.md, .markdown): Direct text reading

## Technical Details

### NLP Models Used

- **Summarization**: Facebook BART (facebook/bart-large-cnn)
- **Classification**: Facebook BART MNLI (facebook/bart-large-mnli)
- **Sentiment Analysis**: Hugging Face default sentiment model
- **Fallback Methods**: NLTK-based extractive summarization when transformers are unavailable

### Key Technologies

- **Streamlit**: Web framework
- **Transformers**: Hugging Face models for NLP tasks
- **NLTK**: Natural language processing toolkit
- **PyPDF2**: PDF text extraction
- **python-docx**: Word document processing

## Project Structure

```
Text-Summarization-App/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .gitignore            # Git ignore rules
└── samples/              # Sample documents for testing (optional)
```

## Performance Considerations

- **File Size Limit**: Works best with files under 10MB
- **Processing Time**: Depends on document length and model availability
  - Small files (< 1000 words): < 10 seconds
  - Medium files (1000-5000 words): 10-30 seconds
  - Large files (> 5000 words): 30-60 seconds
- **Model Loading**: First run downloads models (~2GB), subsequent runs use cached models

## Troubleshooting

### Common Issues

**"PyPDF2 not installed" error**
```bash
pip install PyPDF2
```

**"python-docx not installed" error**
```bash
pip install python-docx
```

**Models fail to load**
- Ensure you have sufficient RAM (8GB+ recommended)
- Check internet connection for first-time model downloads
- The app will fall back to extractive summarization if transformers fail

**Empty or garbled text extraction**
- Ensure PDF is text-based, not scanned images
- Check file encoding for text files
- Try re-saving the document in a standard format

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Future Enhancements

- [ ] Multi-file batch processing
- [ ] Support for scanned PDFs (OCR)
- [ ] Multiple language support
- [ ] Custom model fine-tuning
- [ ] API integration for programmatic access
- [ ] Cloud storage integration (Google Drive, Dropbox)
- [ ] Comparison mode for multiple documents
- [ ] Export to PDF/HTML formats

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for transformer models
- Streamlit for the web framework
- NLTK team for NLP tools
- Open source community for various libraries

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This app processes documents locally or on Streamlit Cloud. No data is stored permanently. Files are deleted after processing for privacy and security.
