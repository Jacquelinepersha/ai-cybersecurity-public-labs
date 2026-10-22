from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

TRAIN_NAME = "UNSW_NB15_training-set.csv"
TEST_NAME = "UNSW_NB15_testing-set.csv"
FEATURES_NAME = "UNSW_NB15_features.csv"

# Compressed copies of the prepared UNSW-NB15 partition ship with this repo
# in data/raw/. If they are missing (for example, a notebook was opened on its
# own), they are downloaded from here automatically.
DATASET_BASE_URL = (
    "https://raw.githubusercontent.com/Jacquelinepersha/"
    "ai-cybersecurity-public-labs/main/data/raw/"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def resolve_data_dir(data_dir=None):
    """Resolve the dataset directory for local use or Colab."""
    if data_dir is not None:
        return Path(data_dir)
    return DEFAULT_DATA_DIR


def _find_or_fetch(data_dir, name, download=True):
    """Return a path to `name` (plain or .gz), downloading the .gz if needed."""
    data_dir = Path(data_dir)
    for candidate in (data_dir / name, data_dir / f"{name}.gz"):
        if candidate.exists():
            return candidate

    if not download:
        return None

    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / f"{name}.gz"
    url = DATASET_BASE_URL + target.name
    print(f"Downloading {target.name} ...")
    try:
        urlretrieve(url, target)
    except Exception as exc:
        if target.exists():
            target.unlink()
        raise FileNotFoundError(
            f"Could not find or download {name}.\n"
            f"Tried: {data_dir / name}, {target}, and {url}\n"
            "Check your internet connection, or download the prepared CSV files "
            "from https://research.unsw.edu.au/projects/unsw-nb15-dataset "
            f"into {data_dir}."
        ) from exc
    return target


def load_unsw(data_dir=None):
    """Load the prepared UNSW-NB15 training and testing sets.

    Works with no setup: uses data/raw/ (plain .csv or .csv.gz) and downloads
    the compressed files automatically if they are not there.
    """
    data_dir = resolve_data_dir(data_dir)
    train_path = _find_or_fetch(data_dir, TRAIN_NAME)
    test_path = _find_or_fetch(data_dir, TEST_NAME)
    return pd.read_csv(train_path), pd.read_csv(test_path)


def load_feature_dictionary(data_dir=None):
    """Load the optional UNSW-NB15 feature-description CSV."""
    data_dir = resolve_data_dir(data_dir)
    path = _find_or_fetch(data_dir, FEATURES_NAME, download=False)
    if path is None:
        return None
    return pd.read_csv(path, encoding="latin1")
