import logging
import pytesseract
import fitz
from PIL import Image
from src.config import OCR_LANG  # добавьте в config.py: OCR_LANG = "eng+rus"

logger = logging.getLogger(__name__)

def ocr_page(pdf_path: str, page_num: int, dpi: int = 300, lang: str = None) -> str:
    """
    Выполняет OCR на указанной странице PDF.
    Возвращает распознанный текст или пустую строку при ошибке.
    """
    if lang is None:
        try:
            from src.config import OCR_LANG
            lang = OCR_LANG
        except ImportError:
            lang = "eng+rus"

    doc = None
    try:
        # Проверка наличия tesseract (опционально)
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            logger.error(f"Tesseract не найден или не настроен: {e}")
            return ""

        doc = fitz.open(pdf_path)
        if page_num < 1 or page_num > len(doc):
            logger.error(f"Некорректный номер страницы: {page_num}")
            return ""

        page = doc[page_num - 1]
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=matrix)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        text = pytesseract.image_to_string(img, lang=lang)
        return text.strip()
    except Exception as e:
        logger.error(f"Ошибка OCR на странице {page_num} файла {pdf_path}: {e}")
        return ""
    finally:
        if doc:
            doc.close()