"""Build tidy league tables from the raw cache: python scripts/build_tidy.py [nba|nfl|all]"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sq.data import nba  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("nba", "all"):
        out = nba.build_all()
        g = out["games"]
        print(g.groupby("season").agg(n=("game_id", "size"), played=("home_pts", lambda s: s.notna().sum()),
                                      tip_known=("tip_known", "mean"), neutral=("neutral", "sum"),
                                      home_win=("home_pts", lambda s: None)).tail(12))
        print(g[g["season"] == 2026].head(3).T)
        print("player box", out["player_box"].shape, "starter flag non-null by season:")
        pb = out["player_box"]
        print(pb.groupby("season")["starter"].apply(lambda s: s.notna().mean()).tail(14))
        print("injuries", out["injuries"].shape)
    if which in ("nfl", "all"):
        from sq.data import nfl  # noqa: E402

        out = nfl.build_all()
        print({k: v.shape for k, v in out.items()})
