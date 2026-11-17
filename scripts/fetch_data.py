#!/usr/bin/env python3
"""Download and verify the UCI online-shoppers dataset."""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

SOURCE_URL = (
    "https://archive.ics.uci.edu/static/public/468/"
    "online+shoppers+purchasing+intention+dataset.zip"
)
EXPECTED_SHA256 = "b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9"
DESTINATION = Path(__file__).resolve().parents[1] / "data" / "raw" / "online_shoppers_intention.csv"


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "purchase-intent-mlops/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def extract(payload: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        name = next(name for name in archive.namelist() if name.endswith(".csv"))
        return archive.read(name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = parser.parse_args()

    if DESTINATION.exists() and not args.force:
        print(f"{DESTINATION} already exists; pass --force to refresh")
        return 0

    print(f"downloading {SOURCE_URL}")
    csv_bytes = extract(download(SOURCE_URL))
    digest = hashlib.sha256(csv_bytes).hexdigest()
    if digest != EXPECTED_SHA256:
        print(f"checksum mismatch: {digest}", file=sys.stderr)
        return 1

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_bytes(csv_bytes)
    print(f"wrote {DESTINATION} ({len(csv_bytes)} bytes, sha256 verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
