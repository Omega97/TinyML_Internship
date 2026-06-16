"""
Download Lichess datasets and save them to the correct data folders.
"""
import kagglehub
import pandas as pd
import sys
from pathlib import Path

# Runtime path hack for direct runs (consistent with other scripts)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tinymlinternship.config.settings import (
    PROJECT_ROOT,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    LICHESS_CSV
)


def ensure_dirs():
    """Create necessary directories if they don't exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Data directories ready under: {PROJECT_ROOT}")


def download_lichess_dataset():
    """Download the small chess dataset from Kaggle."""
    print("Downloading Kaggle chess dataset...")

    path = kagglehub.dataset_download("datasnaek/chess")
    print(f"Dataset downloaded to: {path}")

    # Source file
    source_csv = Path(path) / "games.csv"

    # Destination
    dest_csv = RAW_DATA_DIR / "games.csv"

    if source_csv.exists():
        import shutil
        shutil.copy(source_csv, dest_csv)
        print(f"✅ Copied dataset to: {dest_csv}")
    else:
        print("⚠️ Could not find games.csv in downloaded dataset.")


def main():
    ensure_dirs()
    download_lichess_dataset()

    # Quick check
    if LICHESS_CSV.exists():
        df = pd.read_csv(LICHESS_CSV)
        print(f"✅ Dataset loaded successfully! Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
