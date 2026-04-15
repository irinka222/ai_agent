from src.ingestion.pdf_parser import parse_pdf

pdf_path = "data/raw/collection_1/paper1.pdf"

blocks = parse_pdf(pdf_path)

for block in blocks:
    print("\n")
    print(f"PAGE: {block.page} | SOURCE: {block.source}")
    print("\n")

    print(block.content[:5000])