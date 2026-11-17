"""Generate dashboard/index.html (self-contained; data embedded): python scripts/build_dashboard.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sq.config import ROOT  # noqa: E402
from sq.research.dashboard_data import collect  # noqa: E402

def clean(x):
    """Replace NaN/inf with None so the embedded JSON is valid."""
    import math

    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


if __name__ == "__main__":
    data = clean(collect())
    tpl = (ROOT / "sq" / "research" / "dashboard_template.html").read_text(encoding="utf-8")
    html = tpl.replace("__DATA__", json.dumps(data, separators=(",", ":"), allow_nan=False))
    out = ROOT / "dashboard" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({len(html) / 1e6:.2f} MB)")
