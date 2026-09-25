from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The raw dataset files are too large to live in the code repo and stay on
# the original Google-Drive-synced folder (the repo itself was moved off
# Drive to a local path after repeated git lock corruption there — see the
# SDD ledger). This is a single-machine research repo, so a hardcoded
# absolute path is the simplest correct thing; update it if the data moves.
DATA_SOURCE_DIR = Path(
    r"G:\My Drive\UNEB\PPGMSB\MarcosProducaoCientifica\2026\quantumStudies\quantumTest"
)

RAW_CSV_PATH = DATA_SOURCE_DIR / "pq_v2x_realistic.csv"
RAW_H5_PATH = DATA_SOURCE_DIR / "pq_v2x_dataset.h5"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CSV_CORPUS_DIR = PROCESSED_DIR / "csv_corpus"
H5_CORPUS_DIR = PROCESSED_DIR / "h5_corpus"

RESULTS_DIR = PROJECT_ROOT / "results"

SEED = 42
