import os

GROBID_URL = os.getenv("GROBID_URL", "http://localhost:8070/api/processFulltextDocument")
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "800"))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "1"))
OCR_ENABLED = os.getenv("OCR_ENABLED", "false").lower() == "true"
OCR_LANG = os.getenv("OCR_LANG", "eng+rus")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MATHPIX_APP_ID = os.getenv("MATHPIX_APP_ID", "")
MATHPIX_APP_KEY = os.getenv("MATHPIX_APP_KEY", "")
INCLUDE_REFERENCES = os.getenv("INCLUDE_REFERENCES", "false").lower() == "true"