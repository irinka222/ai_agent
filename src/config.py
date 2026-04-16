import os

GROBID_URL = os.getenv("GROBID_URL", "http://localhost:8070/api/processFulltextDocument")
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "800"))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "1"))
OCR_ENABLED = os.getenv("OCR_ENABLED", "false").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Mathpix (опционально)
MATHPIX_APP_ID = os.getenv("MATHPIX_APP_ID", "")
MATHPIX_APP_KEY = os.getenv("MATHPIX_APP_KEY", "")