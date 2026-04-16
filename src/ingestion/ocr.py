import pytesseract
import fitz
from PIL import Image


def ocr_page(pdf_path: str, page_num: int, dpi: int = 300) -> str:
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]

    # увеличиваем DPI → лучше OCR
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=matrix)

    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    text = pytesseract.image_to_string(
        img,
        lang="eng+rus"
    )

    return text.strip()