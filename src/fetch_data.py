"""Download and extract the UIUC Propeller Database.

The raw archive is not version-controlled (see .gitignore), so this script is
how you reproduce data/raw/ from scratch:

    python src/fetch_data.py

Source: https://m-selig.ae.illinois.edu/props/propDB.html
"""

import hashlib
import sys
import zipfile
from pathlib import Path

import requests

URL = "https://m-selig.ae.illinois.edu/props/download/UIUC-propDB.zip"
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
ARCHIVE = RAW / "UIUC-propDB.zip"


def download(url: str, dest: Path) -> None:
    """Stream `url` to `dest`, printing progress."""
    if dest.exists():
        print(f"{dest.name} already present ({dest.stat().st_size / 1e6:.1f} MB), skipping download.")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB  ({pct:5.1f}%)", end="")
    print()


def extract(archive: Path, dest: Path) -> None:
    """Unzip `archive` into `dest`."""
    print(f"Extracting {archive.name} -> {dest}")
    with zipfile.ZipFile(archive) as z:
        z.extractall(dest)
    print(f"  {sum(1 for _ in dest.rglob('*') if _.is_file())} files on disk.")


def main() -> int:
    download(URL, ARCHIVE)
    sha = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    print(f"sha256: {sha}")
    extract(ARCHIVE, RAW)
    return 0


if __name__ == "__main__":
    sys.exit(main())
