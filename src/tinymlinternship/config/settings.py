from pathlib import Path


# Root of the project (one level above src/)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.absolute()

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# SARDINE pipeline artifacts (active)
SARDINE_MODELS_DIR = PROJECT_ROOT / "models"
TEACHER_DIR = SARDINE_MODELS_DIR / "teacher"
TEACHER_MANIFEST = TEACHER_DIR / "manifest.json"
LC0_BINARY = TEACHER_DIR / "lc0" / "lc0.exe"
LC0_NETWORKS_DIR = TEACHER_DIR / "networks"
LC0_NETWORK_FAST = TEACHER_DIR / "lc0" / "791556.pb.gz"
LC0_NETWORK_T1_256 = LC0_NETWORKS_DIR / "t1-256x10-distilled-swa-2432500.pb.gz"
LC0_NETWORK_BT4 = LC0_NETWORKS_DIR / "BT4-1024x15x32h-swa-6147500.pb.gz"
LC0_NETWORK_DEFAULT = LC0_NETWORK_FAST

LC0_NETWORK_PRESETS: dict[str, Path] = {
    "fast": LC0_NETWORK_FAST,
    "791556": LC0_NETWORK_FAST,
    "t1-256": LC0_NETWORK_T1_256,
    "bt4": LC0_NETWORK_BT4,
}

HF_TEACHER_DIR = TEACHER_DIR / "hf"
CHESS_LITE_WEIGHTS = HF_TEACHER_DIR / "chess_lite" / "chess_lite.pth"
ARTORIA_SMALL_CKPT = HF_TEACHER_DIR / "artoria-zero" / "small" / "checkpoint.pt"
ARTORIA_SMALL_CONFIG = HF_TEACHER_DIR / "artoria-zero" / "small" / "config.json"

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
CHESSBENCH_RAW_DIR = RAW_DATA_DIR / "chessbench"
CHESSBENCH_PROCESSED_DIR = PROCESSED_DATA_DIR / "chessbench"

# You can also make it environment-aware
import os
ENV = os.getenv("ENV", "development")  # development / production

# Optional: Allow override via environment variable
DATA_DIR = Path(os.getenv("DATA_DIR", DATA_DIR))
