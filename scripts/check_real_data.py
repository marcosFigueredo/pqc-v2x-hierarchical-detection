# scripts/check_real_data.py
import json

import pandas as pd

from src.config import RAW_CSV_PATH, RAW_H5_PATH
from src.data_prep import check_corpus_identity, check_session_leakage

if __name__ == "__main__":
    identity = check_corpus_identity(RAW_CSV_PATH, RAW_H5_PATH, n=20000)
    print("Corpus identity check:")
    print(json.dumps(identity, indent=2))

    csv_sample = pd.read_csv(RAW_CSV_PATH, usecols=["session_id"], nrows=500_000)
    session_leakage = check_session_leakage(csv_sample)
    print("\nSession leakage check (first 500k CSV rows):")
    print(json.dumps(session_leakage, indent=2))
