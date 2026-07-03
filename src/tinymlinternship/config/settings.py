from pathlib import Path


# Root of the project (one level above src/)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.absolute()

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# SARDINE pipeline artifacts (active)
SARDINE_MODELS_DIR = PROJECT_ROOT / "models"

# Pre-SARDINE archive (legacy export pipeline)
LEGACY_DIR = PROJECT_ROOT / "legacy" / "pre-sardine"
LEGACY_MODELS_DIR = LEGACY_DIR / "models"
CHECKPOINTS_DIR = LEGACY_MODELS_DIR / "checkpoints"
EXPORTED_DIR = LEGACY_MODELS_DIR / "exported"
ARDUINO_DIR = LEGACY_MODELS_DIR / "arduino"
ARDUINO_MODELS_DIR = ARDUINO_DIR / "models"
WIO_SKETCH_DIR = LEGACY_DIR / "Arduino" / "Wio_TinyValueTest"

# Example specific files
LICHESS_CSV = RAW_DATA_DIR / "games.csv"
LC0_RAW_DIR = RAW_DATA_DIR / "lc0"
LC0_TARS_DIR = LC0_RAW_DIR / "tars"
LC0_CHUNKS_DIR = LC0_RAW_DIR / "chunks"
LC0_MANIFEST = LC0_RAW_DIR / "manifest.json"
LC0_PROCESSED_DIR = PROCESSED_DATA_DIR / "lc0"

# You can also make it environment-aware
import os
ENV = os.getenv("ENV", "development")  # development / production

# Optional: Allow override via environment variable
DATA_DIR = Path(os.getenv("DATA_DIR", DATA_DIR))
