"""
Direct Kaggle download script — bypasses the broken CLI auth.
Usage:
    python scripts/download_kaggle.py --token KGAT_xxxxx
"""
import argparse
import os
import zipfile
import requests
from pathlib import Path

COMPETITION = "march-machine-learning-mania-2025"
OUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "kaggle"

# Files we actually need
TARGET_FILES = [
    "MTeams.csv",
    "MNCAATourneySeeds.csv",
    "MNCAATourneyCompactResults.csv",
    "MRegularSeasonCompactResults.csv",
    "MNCAATourneyDetailedResults.csv",
    "MRegularSeasonDetailedResults.csv",
]

def download_file(token: str, filename: str, out_dir: Path) -> bool:
    dest = out_dir / filename
    if dest.exists():
        print(f"  SKIP  {filename} (already exists)")
        return True

    url = f"https://www.kaggle.com/api/v1/competitions/data/download/{COMPETITION}/{filename}"
    headers = {"Authorization": f"Bearer {token}"}

    print(f"  GET   {filename} ...", end="", flush=True)
    r = requests.get(url, headers=headers, stream=True, timeout=60)

    if r.status_code == 200:
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
        size_kb = dest.stat().st_size // 1024
        print(f" OK ({size_kb} KB)")
        return True
    elif r.status_code == 403:
        print(f" FORBIDDEN — have you accepted competition rules?")
        print(f"    Go to: https://www.kaggle.com/competitions/{COMPETITION}/rules")
        return False
    elif r.status_code == 404:
        print(f" NOT FOUND (file may not exist in this competition)")
        return False
    else:
        print(f" ERROR {r.status_code}: {r.text[:200]}")
        return False


def try_bulk_download(token: str, out_dir: Path) -> bool:
    """Try downloading the full competition zip."""
    url = f"https://www.kaggle.com/api/v1/competitions/data/download-all/{COMPETITION}"
    headers = {"Authorization": f"Bearer {token}"}

    print(f"\nAttempting bulk download of all competition files...")
    r = requests.get(url, headers=headers, stream=True, timeout=120)

    if r.status_code == 200:
        zip_path = out_dir / f"{COMPETITION}.zip"
        print(f"  Downloading zip...", end="", flush=True)
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
        size_mb = zip_path.stat().st_size // (1024 * 1024)
        print(f" OK ({size_mb} MB)")

        print(f"  Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(out_dir)
        zip_path.unlink()
        print(f"  Extracted to {out_dir}")
        return True
    else:
        print(f"  Bulk download failed ({r.status_code}) — trying individual files.")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="Kaggle API token (KGAT_...)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_DIR}\n")

    # Try bulk first
    if not try_bulk_download(args.token, OUT_DIR):
        # Fall back to individual files
        print("\nDownloading files individually:")
        results = {}
        for f in TARGET_FILES:
            results[f] = download_file(args.token, f, OUT_DIR)

    # Show what we got
    print("\n=== Files in kaggle directory ===")
    for f in sorted(OUT_DIR.iterdir()):
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name:50s} {size_kb:>8} KB")


if __name__ == "__main__":
    main()
