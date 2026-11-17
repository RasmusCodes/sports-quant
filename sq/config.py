"""Paths and global constants."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW_DIR = DATA / "raw"          # untouched downloads + _manifest.jsonl per source
TIDY_DIR = DATA / "tidy"        # normalised parquet tables
DB_PATH = DATA / "db" / "sports.sqlite"
MANUAL_DIR = DATA / "manual"    # files the user downloads by hand (e.g. sportsbook odds exports)
OUT = ROOT / "outputs"

# Generic project user agent. Never put personal contact details here.
USER_AGENT = "sports-quant-research/0.1 (personal academic research; polite, cached)"

for _p in (RAW_DIR, TIDY_DIR, DB_PATH.parent, MANUAL_DIR, OUT):
    _p.mkdir(parents=True, exist_ok=True)
