from pathlib import Path


# Root of the project (one level above src/)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.absolute()

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
EXPORTED_DIR = MODELS_DIR / "exported"

# Example specific files
LICHESS_CSV = RAW_DATA_DIR / "games.csv"

# You can also make it environment-aware
import os
ENV = os.getenv("ENV", "development")  # development / production

# Optional: Allow override via environment variable
DATA_DIR = Path(os.getenv("DATA_DIR", DATA_DIR))
