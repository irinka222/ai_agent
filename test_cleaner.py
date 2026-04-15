import sys
sys.path.insert(0, 'src')
from ingestion.grobid_parser import parse_pdf_grobid
from processing.tei_parser import parse_tei
from processing.cleaner import clean_blocks, detect_repeated_lines

pdf_path = "data/raw/collection_1/0902.4290v1.pdf"

xml = parse_pdf_grobid(pdf_path)
blocks = parse_tei(xml, paper_id="test")

# 1. собираем "страницы" (грубо)
pages = [b.content for b in blocks if hasattr(b, "content")]

# 2. находим мусор
noise_lines = detect_repeated_lines(pages)

# 3. чистим
for b in blocks[:10]:
    text = clean_text(b.content, noise_lines)

    print("\n====================")
    print(text[:500])