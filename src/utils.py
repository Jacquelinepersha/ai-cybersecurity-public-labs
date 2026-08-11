
from pathlib import Path
import random
import numpy as np

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def ensure_results_dir(path="results"):
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output
