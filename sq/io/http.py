"""Polite, cached HTTP fetching with a per-source manifest.

Every network response that is kept is written once to ``data/raw/<source>/`` and
logged to ``_manifest.jsonl`` with the URL, UTC fetch time, byte count and sha256,
so every number downstream can be traced to a dated raw file. Re-runs read the
cache instead of the network unless ``refresh=True``.

Rules enforced here (project ground rules): generic user agent, a minimum
interval between requests per source, exponential backoff that honours
Retry-After on 429/5xx, and no attempt to get around blocks (a 401/403 is
raised, not retried with other headers).
"""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

from ..config import RAW_DIR, USER_AGENT


class FetchError(RuntimeError):
    pass


def utcnow_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Fetcher:
    def __init__(self, source: str, min_interval: float = 0.25, timeout: float = 60.0,
                 max_retries: int = 6):
        self.source = source
        self.dir = RAW_DIR / source
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self.dir / "_manifest.jsonl"
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.min_interval = min_interval
        self.timeout = timeout
        self.max_retries = max_retries
        self._last = 0.0
        self.n_network = 0
        self.n_cache = 0

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def full_url(url: str, params: dict | None) -> str:
        if not params:
            return url
        items = sorted((k, v) for k, v in params.items() if v is not None)
        return url + "?" + urlencode(items)

    def _wait(self) -> None:
        gap = time.monotonic() - self._last
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last = time.monotonic()

    def _log(self, full: str, path: Path, content: bytes, status: int) -> None:
        rec = {"url": full, "fetched_at_utc": utcnow_iso(), "status": status, "bytes": len(content),
               "sha256": hashlib.sha256(content).hexdigest(), "path": str(path.relative_to(self.dir))}
        with self.manifest.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    def _request(self, url: str, params: dict | None) -> requests.Response:
        delay = 2.0
        last_exc: Exception | None = None
        for _ in range(self.max_retries):
            self._wait()
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                ra = r.headers.get("Retry-After")
                wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else delay
                time.sleep(min(wait, 300))
                delay = min(delay * 2, 120)
                last_exc = FetchError(f"{r.status_code} for {r.url}")
                continue
            self.n_network += 1
            return r
        raise FetchError(f"giving up on {url} {params}: {last_exc}")

    # --------------------------------------------------------------------- API
    def get_json(self, url: str, params: dict | None = None, *, name: str | None = None,
                 refresh: bool = False, allow_404: bool = False) -> Any:
        """GET JSON with a gzip cache file named ``name`` (default: hash of the URL)."""
        full = self.full_url(url, params)
        name = name or hashlib.sha1(full.encode()).hexdigest()[:24]
        path = self.dir / f"{name}.json.gz"
        if path.exists() and not refresh:
            self.n_cache += 1
            return json.loads(gzip.decompress(path.read_bytes()))
        r = self._request(url, params)
        if r.status_code == 404 and allow_404:
            return None
        if r.status_code != 200:
            raise FetchError(f"HTTP {r.status_code} for {full}: {r.text[:200]}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(r.content))
        self._log(full, path, r.content, r.status_code)
        return r.json()

    def get_file(self, url: str, rel_path: str, *, refresh: bool = False,
                 allow_404: bool = False) -> Path | None:
        """Download a binary/text file once to ``data/raw/<source>/<rel_path>``."""
        path = self.dir / rel_path
        if path.exists() and not refresh:
            self.n_cache += 1
            return path
        r = self._request(url, None)
        if r.status_code == 404 and allow_404:
            return None
        if r.status_code != 200:
            raise FetchError(f"HTTP {r.status_code} for {url}")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(r.content)
        tmp.replace(path)
        self._log(url, path, r.content, r.status_code)
        return path
