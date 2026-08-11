
from pathlib import Path
import pandas as pd

TRAIN_NAME = "UNSW_NB15_training-set.csv"
TEST_NAME = "UNSW_NB15_testing-set.csv"
FEATURES_NAME = "UNSW_NB15_features.csv"

def resolve_data_dir(data_dir=None):
    """Resolve a dataset directory for local use or Colab."""
    if data_dir is not None:
        return Path(data_dir)

    candidates = [
        Path("data/raw"),
        Path("../data/raw"),
        Path("/content/ai-cybersecurity-masterclass/data/raw"),
        Path("/content/drive/MyDrive/AI_Cybersecurity/data/raw"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return Path("data/raw")

def load_unsw(data_dir=None):
    """Load the prepared UNSW-NB15 training and testing CSV files."""
    data_dir = resolve_data_dir(data_dir)

    train_path = data_dir / TRAIN_NAME
    test_path = data_dir / TEST_NAME

    missing = [path for path in (train_path, test_path) if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Missing UNSW-NB15 files:\n"
            f"{formatted}\n\n"
            "Download the official prepared training and testing CSV files "
            "and place them in data/raw/, or pass the correct data directory."
        )

    return pd.read_csv(train_path), pd.read_csv(test_path)

def load_feature_dictionary(data_dir=None):
    """Load the optional UNSW-NB15 feature-description CSV."""
    data_dir = resolve_data_dir(data_dir)
    path = data_dir / FEATURES_NAME

    if not path.exists():
        return None

    return pd.read_csv(path, encoding="latin1")
