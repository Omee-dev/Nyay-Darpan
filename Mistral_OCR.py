import os
import base64
from mistralai import Mistral
# The models import for OCRResponse is valid
from mistralai.models import OCRResponse  
from dotenv import load_dotenv
load_dotenv()
def mistral_ocr(pdf_path: str, include_images: bool = False) -> dict[int, str]:
    """
    OCR a PDF using the official Mistral client library.
    Returns a dict mapping page numbers (1-based) to markdown text for that page.
    """

    # Load API key
    api_key = os.getenv("MISTRAL_OCR_KEY")
    if not api_key:
        raise ValueError("Please set the environment variable MISTRAL_API_KEY")
    client = Mistral(api_key=api_key)

    # Read PDF and encode base64
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")

    # Create document object 
    document = {
        "type": "document_url",
        # “document_url” can also accept a base64-encoded PDF, prefixed correctly
        "document_url": f"data:application/pdf;base64,{pdf_b64}"
    }

    # Call OCR
    resp: OCRResponse = client.ocr.process(
        model="mistral-ocr-latest",
        document=document,
        include_image_base64=include_images
    )

    # Build page-wise markdown
    page_texts: dict[int, str] = {}
    for idx, page in enumerate(resp.pages, start=1):
        markdown = page.markdown or ""
        page_texts[idx] = markdown

    return page_texts

# Example usage
if __name__ == "__main__":
    pdf_path = r".\document.pdf"
    result = mistral_ocr(pdf_path, include_images=False)
    for pageno, md in result.items():
        print(f"--- Page {pageno} ---")
        print(md)
        print()

