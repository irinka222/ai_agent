from src.ingestion.grobid_parser import parse_pdf_grobid
from src.processing.tei_parser import parse_tei
from pathlib import Path

pdf_path = "data/raw/collection_1/0902.4290v1.pdf"

paper_id = Path(pdf_path).stem

xml = parse_pdf_grobid(pdf_path)

blocks = parse_tei(xml, paper_id)

for b in blocks[:10]:
    print("\n" + "="*60)
    print(f"TYPE: {b.block_type}")
    print(b.content[:500])