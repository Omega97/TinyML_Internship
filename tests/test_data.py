"""
Tests for data loading and basic dataset validation.
"""
import pandas as pd
from src.tinymlinternship.config.settings import RAW_DATA_DIR, LICHESS_CSV


def test_data_files_exist():
    """Check that the raw dataset was downloaded."""
    assert RAW_DATA_DIR.exists(), f"Data directory not found: {RAW_DATA_DIR}"
    assert LICHESS_CSV.exists(), f"Dataset file not found: {LICHESS_CSV}"
    print(f"✅ Dataset found at: {LICHESS_CSV}")


def test_load_dataset_head():
    """Load the dataset and display the first few rows."""
    assert LICHESS_CSV.exists(), "Dataset file does not exist. Run download_data.py first."

    df = pd.read_csv(LICHESS_CSV)

    print("\n" + "=" * 80)
    print("DATASET HEAD (First 5 rows)")
    print("=" * 80)
    print(df.head())
    print("\n" + "=" * 80)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("=" * 80)

    # Basic assertions
    assert len(df) > 0, "Dataset is empty"
    assert 'moves' in df.columns, "'moves' column not found"
    assert 'winner' in df.columns, "'winner' column not found"

    print("✅ Basic data validation passed!")


if __name__ == "__main__":
    test_data_files_exist()
    test_load_dataset_head()
