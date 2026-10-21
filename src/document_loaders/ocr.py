import os 
import fitz  
from PIL import Image
import pytesseract
from langchain_core.documents import Document

"""Number of text wordings after which OCR will run"""

MIN_TEXT_CHARS = 20
OCR_DPI = 200 #accuracy

def pdf_load_ocr(pdf_path: str) -> list[Document]:
    documents: list[Document] = []
    pdf = fitz.open(pdf_path)

    for page_num, page in enumerate(pdf, start = 1):
        text = page.get_text("text").strip()

        if len(text) < MIN_TEXT_CHARS:
            pix = page.get_pixmap(dpi = OCR_DPI)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            text = pytesseract.image_to_string(img).strip()


        if text:
            documents.append(
                Document(
                    page_content = text,
                    metadata = {"page": page_num, "source": os.path.basename(pdf_path)},
                )
            )

        pdf.close()
        return documents