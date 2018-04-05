import pandas as pd
from pathlib import Path


def batch_load_csv(folder, *files):
    input_dir = Path(folder)
    return (pd.read_csv(input_dir / f) for f in files)
