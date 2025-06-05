import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db2")
SQLITE_DB_PATH = os.path.join(BASE_DIR, "db/jobs.db")
MAX_RETRIES = 2