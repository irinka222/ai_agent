from src.ingestion.grobid_parser import parse_pdf_grobid
from pathlib import Path

pdf_folder = Path("data/raw/collection_1")
pdf_files = list(pdf_folder.glob("*.pdf"))

if pdf_files:
    pdf_path = str(pdf_files[0])
    print(f"Обрабатываю: {pdf_path}")
    xml = parse_pdf_grobid(pdf_path)
    print("\n--- XML START ---\n")
    print(xml[:2000])
else:
    print("Нет PDF-файлов в папке")