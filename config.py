from pathlib import Path

# Root directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Database directory and other data directories
DATA_DIR = BASE_DIR / "data"
SHOTS_DIR = DATA_DIR / "extracted_shots"
VIDEOS_DIR = DATA_DIR / "videos_ingested"
DB_DIR = BASE_DIR / "databases"

# Make sure the directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
SHOTS_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)

# database paths

SQLITE_DB_PATH = DB_DIR / "sqlite" / "metadata.db"

CHROMA_PERSIST_DIR = str(DB_DIR / "chroma_db")

# AI model configuration
MODEL_NAME = "ViT-B-16-SigLIP"
PRETRAINED_SOURCE = "webLI"