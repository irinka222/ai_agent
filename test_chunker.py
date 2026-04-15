import sys

sys.path.insert(0, 'src')

from ingestion.grobid_parser import parse_pdf_grobid
from processing.tei_parser import parse_tei
from processing.chunker import chunk_blocks

pdf_path = "data/raw/collection_1/0902.4290v1.pdf"

# парсим PDF через GROBID
xml = parse_pdf_grobid(pdf_path)

# ОБЯЗАТЕЛЬНО передаём paper_id
blocks = parse_tei(xml, paper_id="test_paper")

# делаем чанки
chunks = chunk_blocks(blocks)

print(f"\nBlocks: {len(blocks)}")
print(f"Chunks: {len(chunks)}")

# вывод первых чанков
for i, chunk in enumerate(chunks[:5]):
    print("\n" + "=" * 60)
    print(f"CHUNK {i+1}")
    print("=" * 60)
    print(chunk[:500])