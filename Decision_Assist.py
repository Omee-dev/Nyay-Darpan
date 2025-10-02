from Mistral_OCR import mistral_ocr
from flask import Flask, request, render_template, jsonify, session, Blueprint
import os
import fitz 
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from prompts import system_prompts
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
import google.generativeai as genai  # Gemini client
from Retrieval import (
    find_similar_cases,
    get_top5_citations,
    get_top5_citations_bm25,
    get_top5_sectoral_chunks,
    get_top5_sectoral_chunks_bm25,
    case_data
)
from llama_handler import llama_extract, DEFAULT_LLAMA_MODEL

app = Flask(__name__)
bp = Blueprint('decision_tool', __name__, url_prefix='/decision-assist-tool')

# Load env vars
load_dotenv()
import logging


# Flask app setup
app = Flask(__name__)
app.secret_key = 'supersecretkey'

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
LOG_FILE = os.path.join(os.path.abspath(UPLOAD_FOLDER), "upload_log.txt")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not set in environment")
genai.configure(api_key=api_key)

# Default model
default_model = "gemini-2.5-flash"

# Load semantic model and case DB
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
case_data = pd.read_csv('./Data/output_ss3.csv')

# ------------------ File Handling ------------------
def extract_text_from_file(file_path):
    if file_path.endswith('.pdf'):
        doc = fitz.open(file_path)
        text = "".join(page.get_text() for page in doc)
        return text
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif file_path.endswith('.docx'):
        from docx import Document
        doc = Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs)
    return ""


# ------------------ Gemini Extract ------------------
def gemini_extract(case_text, system_prompt, model_name="gemini-2.5-flash"):
    """Calls Gemini API to extract structured parts of the case"""
    # Ensure Gemini models always have the correct "models/" prefix
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"

    model = genai.GenerativeModel(model_name)
    response = model.generate_content(system_prompt + "\n\n" + case_text)
    return response.text.strip() if response.text else ""


def extract_sector_code(summary: dict) -> int:
    # Try Sector field first
    sector_text = summary.get("Sector", "")
    match = re.search(r'\d+', sector_text)
    if match:
        return int(match.group())
    return 999  # Default if nothing found

# ------------------ Routes ------------------
# ------------------ Routes ------------------
@bp.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        upload_mode = request.form.get('upload_mode', 'normal') 
        case_text = request.form.get('case_text', '').strip()
        selected_model = request.form.get('model', default_model)

        filename = None
        uploaded_files = request.files.getlist('case_file')
        extracted_texts = []
        filenames_combined = []  # <-- store all filenames

        for uploaded_file in uploaded_files:
            if uploaded_file and uploaded_file.filename != '':
                filename = secure_filename(uploaded_file.filename)
                filenames_combined.append(filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                uploaded_file.save(file_path)

                if upload_mode == "ocr":
                    ocr_output = mistral_ocr(file_path)
                    extracted_texts.append(f"## {filename}\n{' '.join(ocr_output.values())}\n")
                else:
                    file_text = extract_text_from_file(file_path)
                    extracted_texts.append(f"## {filename}\n{file_text}\n")

        # Append manual input if present
        if upload_mode == "normal" and case_text:
            extracted_texts.append(f"## manual_input.txt\n{case_text}\n")
            filenames_combined.append("manual_input.txt")

        # Combine everything into one string
        case_text = "\n".join(extracted_texts).strip()

        # Store all filenames in session as string
        session['filename'] = "\n".join(filenames_combined)
        session['case_text'] = case_text
        session['model'] = selected_model
        
                # ------------------ LOG to file ------------------
        try:
            with open(LOG_FILE, 'a', encoding='utf-8') as f:
                f.write("----- New Upload -----\n")
                f.write(f"Uploaded Files:\n{session['filename']}\n\n")
                f.write(f"Extracted Text:\n{case_text}\n")
                f.write("----------------------\n\n")
        except Exception as e:
            app.logger.warning(f"Failed to write log: {e}")
        if "llama" in selected_model.lower():
            summary = {
                part: llama_extract(case_text, prompt, selected_model)
                for part, prompt in system_prompts.items()
            }
        else:
            summary = {
                part: gemini_extract(case_text, prompt, selected_model)
                for part, prompt in system_prompts.items()
            }

        # Extract sector code from Reliefs if possible
        sector_code = extract_sector_code(summary)

        overview_text = summary.get("Overview", "")

        # Similar cases
        similar_cases = find_similar_cases(overview_text, sector_code)
        summary["Similar Cases"] = similar_cases

        # Sectional citations
        similar_citations = get_top5_citations(overview_text, embed_model)
        bm25_citations = get_top5_citations_bm25(overview_text)

        # Sectoral citations
        sectoral_chunks = get_top5_sectoral_chunks(overview_text, sector_code, k=5)
        sectoral_chunks_bm25 = get_top5_sectoral_chunks_bm25(overview_text, sector_code, k=5)

        return render_template(
            'summary.html',
            summary=summary,
            filename=session['filename'],
            similar_citations=similar_citations,
            bm25_citations=bm25_citations,
            sectoral_chunks=sectoral_chunks,
            sectoral_chunks_bm25=sectoral_chunks_bm25
        )

    return render_template('index2.html')



@bp.route('/ocr', methods=['POST'])
def ocr_page():
    uploaded_file = request.files.get('case_file')
    # Accept 'on' (checkbox) or explicit 'true'/'1'
    include_images = request.form.get('include_images') in ('on', 'true', '1', 'yes')

    if not uploaded_file or uploaded_file.filename == '':
        return jsonify({'error': 'No file uploaded'}), 400

    if not uploaded_file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are accepted for OCR'}), 400

    filename = secure_filename(uploaded_file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        uploaded_file.save(file_path)
    except Exception as e:
        app.logger.exception("Failed to save uploaded file")
        return jsonify({'error': f'Failed to save file: {str(e)}'}), 500

    try:
        # Run OCR - this is the likely place for exceptions
        ocr_text = mistral_ocr(file_path, include_images)
        if ocr_text is None:
            ocr_text = ""
    except Exception as e:
        app.logger.exception("Mistral OCR processing failed")
        return jsonify({'error': f'OCR processing error: {str(e)}'}), 500

    # Optionally delete file after processing (uncomment if desired)
    # try:
    #     os.remove(file_path)
    # except Exception:
    #     pass

    return jsonify({'filename': filename, 'ocr_text': ocr_text}), 200

@bp.route('/regenerate', methods=['POST'])
def regenerate():
    data = request.get_json()
    section = data.get('section')
    case_text = session.get('case_text')
    model = session.get('model', default_model)

    if not section or not case_text:
        return jsonify({'error': 'Missing section or text'}), 400

    prompt = system_prompts.get(section)
    if not prompt:
        return jsonify({'error': 'Invalid section'}), 400

    result = gemini_extract(case_text, prompt, model)
    return jsonify({'output': result})


@bp.route('/reprocess-model', methods=['POST'])
def reprocess_model():
    data = request.get_json()
    new_model = data.get('model', default_model)
    case_text = session.get('case_text')
    session['model'] = new_model

    if not case_text:
        return jsonify({'error': 'No text found'}), 400

#     summary = {part: gemini_extract(case_text, prompt, new_model)
#                for part, prompt in system_prompts.items()}
    if "llama" in new_model.lower():
        summary = {part: llama_extract(case_text, prompt, new_model)
                   for part, prompt in system_prompts.items()}
    else:
        summary = {part: gemini_extract(case_text, prompt, new_model)
                   for part, prompt in system_prompts.items()}


    reliefs_text = summary.get("Reliefs", "")
    match = re.search(r'\d+', reliefs_text)
    sector_code = int(match.group()) if match else 999

    overview_text = summary.get("Overview", "")
    similar_cases = find_similar_cases(overview_text, sector_code)
    summary["Similar Cases"] = similar_cases

    return jsonify({'summary': summary})


# if __name__ == '__main__':
#     nltk.download('punkt')  
#     app.run(debug=True,port=4000)
app.register_blueprint(bp)
if __name__ == "__main__":
    # must bind to 0.0.0.0 so Docker can expose it
    nltk.download('punkt')
    app.run(host="0.0.0.0", port=5000, debug=False)