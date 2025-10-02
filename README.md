# Decision Assist Tool

A Flask-based web application designed to assist **judges and official decision-makers** in analyzing and summarizing legal case files.  
The tool enables structured extraction, similarity search, and contextual citation retrieval to support efficient and informed decision-making.

---

## ⚖️ Motivation

The **Decision Assist Tool** is intended for use by **judges and official decision-makers**.  
Its purpose is to accelerate the review of lengthy case files by providing:  
- Concise summaries  
- Contextually relevant citations  
- Retrieval of similar past cases  

This ensures faster, more consistent, and better-informed legal decision-making.

---

## ✨ Features

- **File Uploads**: Accepts `.pdf`, `.docx`, `.txt` files or manual text input.
- **OCR**: Uses `Mistral_OCR` for scanned PDFs.
- **Summarization**: 
  - Google Gemini (`gemini-2.5-flash`,...any model)
  - LLaMA Hosted at backend (`llama_3.1_8b`)
  - Add your own summarizer from HF, Groq, etc.
- **Case Retrieval**:
  - SentenceTransformer embeddings
  - BM25 keyword-based ranking
  - Cosine Embeddings ranking
- **Outputs**:
  - Case overview and sector classification
  - Similar cases
  - Relevant citations

---

## 🛠️ Tech Stack

- **Flask** (Python web framework)
- **PyMuPDF (fitz)** for PDF parsing
- **python-docx** for Word docs
- **SentenceTransformers** + **BM25Okapi** for similarity search
- **Google GenerativeAI (Gemini)** for LLM-based extraction
- **Custom LLaMA handler** for local models
- **NLTK** for tokenization

---

## 📂 Project Structure

```
├── Data/                    # Case dataset (CSV files)
├── templates/               # HTML templates (index2.html, summary.html)
├── uploads/                 # Uploaded files
├── Retrieval.py             # Similar case & citation retrieval logic
├── llama_handler.py         # LLaMA backend integration
├── prompts.py               # System prompts for LLMs
├── app.py                   # Main Flask app
```

---

## 🚀 Running the App

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up environment variables
Create a `.env` file:
```bash
GEMINI_API_KEY=your_google_api_key
```

### 3. Run the app
```bash
python app.py
```

By default, the app will start on:
```
http://0.0.0.0:5055/decision-assist-tool/
```

---

## 🌐 URL Prefix

This app is served under the base path:

```
/decision-assist-tool/
```

All routes (including `/ocr`, `/regenerate`, `/reprocess-model`) are available under this prefix.  

Examples:
- Main page → `http://<host>:5055/decision-assist-tool/`
- OCR API → `http://<host>:5055/decision-assist-tool/ocr`
- Regenerate section → `http://<host>:5055/decision-assist-tool/regenerate`

---

## 📝 Logging

Uploaded files and extracted text are logged in:

```
uploads/upload_log.txt
```

---

## ⚠️ Notes

- Ensure **NLTK tokenizer** is available:
  ```python
  import nltk
  nltk.download('punkt')
  ```
- If running behind a reverse proxy (e.g., Nginx/Apache), make sure it passes the `/decision-assist-tool/` prefix correctly to Flask.
- Static files and template URLs must use `url_for` so the prefix is applied automatically.
---
