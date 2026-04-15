from src.ingestion.ocr import ocr_page

# Используем реальный PDF-файл (один из тех, что есть в папке)
pdf_path = "data/raw/collection_1/paper1.pdf"  # или любой другой файл из списка

text = ocr_page(pdf_path, 1)

print("\n--- OCR RESULT ---\n")
print(text[:1000])