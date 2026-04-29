from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer


DATA_DIR = Path("data")
RAW_DATA_PATH = DATA_DIR / "breast_cancer.csv"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    cancer = load_breast_cancer(as_frame=True)
    df = cancer.frame.copy()
    df.to_csv(RAW_DATA_PATH, index=False)

    print(f"Dataset saved to: {RAW_DATA_PATH}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main()

