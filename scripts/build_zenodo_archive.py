"""Build the Zenodo deposit archive for the artifact release.

Packages exactly what the manuscript promises is available for auditing —
code, protocol, seeds, tests, logs, results — and deliberately nothing else.
The raw PQ-V2X releases are not redistributed: they belong to their own
authors and total about 8.7 GB.
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp" / "zenodo_archive.zip"

# Directories copied whole, minus the exclusions below.
INCLUDE_DIRS = ["src", "tests", "scripts", "results", "logs"]
INCLUDE_FILES = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "REPRODUCING.md",
    ".zenodo.json",
    "requirements.txt",
    "pyproject.toml",
    "references.bib",
]
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


def _included(path: Path) -> bool:
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return False
    return path.suffix not in EXCLUDE_SUFFIXES


def main():
    missing = [f for f in INCLUDE_FILES if not (ROOT / f).exists()]
    if missing:
        sys.exit(f"refusing to build an incomplete archive; missing: {missing}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in INCLUDE_FILES:
            archive.write(ROOT / name, name)
            written += 1
        for dirname in INCLUDE_DIRS:
            directory = ROOT / dirname
            if not directory.exists():
                print(f"note: {dirname}/ not present, skipping")
                continue
            for path in sorted(directory.rglob("*")):
                if path.is_file() and _included(path):
                    archive.write(path, str(path.relative_to(ROOT)))
                    written += 1

    size_mb = OUT.stat().st_size / (1024 * 1024)
    print(f"wrote {OUT} ({written} files, {size_mb:.1f} MB)")
    print("Upload this to Zenodo, then put the resulting DOI in the manuscript's")
    print("data availability statement (it currently carries a placeholder).")


if __name__ == "__main__":
    main()
