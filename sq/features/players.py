"""Player-level availability model for the NBA.

For every player we keep exponentially weighted, strictly pre-game estimates of
  v  - production per 36 minutes (Hollinger Game Score), shrunk toward a bench prior
  m  - expected minutes when available
and define the value a team loses when he is out as
  lost_i = max(v_i - v_rep, 0) * m_i / 36        (Game Score above replacement per game)

For each game and team the candidates are players whose latest box-score row was
for that team within its previous 15 games (traded players move with their rows).
Three availability read-outs are produced per team:
  lost_real   realised absences from the game's own box score (DND / NWT / injury /
              rest / suspension comments, or missing from the box). The inactive list is
              public before tip-off, so this is tip-time information - but it is used
              only for history and as the ratings' baseline, never for an earlier
              decision time.
  lost_pit    point-in-time expectation from the official injury report: latest report
              published before the decision time; P(play) = 0 Out, 0.15 Doubtful,
              0.55 Questionable, 0.9 Probable, 1 Available / not listed. Falls back to
              lost_real only when no report exists for the game (pre-2019 seasons).
  lost_base   mean lost_real over the team's previous 10 games - what the team ratings
              have already absorbed. The model feature is lost_pit - lost_base.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np
import pandas as pd

P_PLAY = {"Out": 0.0, "Doubtful": 0.15, "Questionable": 0.55, "Probable": 0.9, "Available": 1.0}
UNAVAILABLE = re.compile(r"injur|illness|rest|personal|suspen|not with team|nwt|dnd|inactive|g league|health|"
                         r"reconditioning|protocol|ineligible|trade", re.I)


def game_score(pb: pd.DataFrame) -> pd.Series:
    return (pb["pts"] + 0.4 * pb["fgm"] - 0.7 * pb["fga"] - 0.4 * (pb["fta"] - pb["ftm"]) + 0.7 * pb["oreb"]
            + 0.3 * pb["dreb"] + pb["stl"] + 0.7 * pb["ast"] + 0.7 * pb["blk"] - 0.4 * pb["pf"] - pb["tov"])


class PlayerState:
    __slots__ = ("team", "last_team_idx", "vn", "vd", "mn", "md", "season")

    def __init__(self):
        self.team, self.last_team_idx, self.vn, self.vd, self.mn, self.md, self.season = None, -10 ** 9, 0.0, 0.0, 0.0, 0.0, None


def availability_features(games: pd.DataFrame, pb: pd.DataFrame, inj: pd.DataFrame | None,
                          decision_offsets_min=(5, 24 * 60), half_life_games=20.0, minutes_half_life=6.0,
                          prior_min=150.0, v0=8.0, v_rep=9.0, window=15, base_n=10) -> pd.DataFrame:
    g = games.sort_values(["tip_utc", "game_id"]).reset_index(drop=True)
    pb = pb.copy()
    pb["gmsc"] = game_score(pb).fillna(0.0)
    pb["person_id"] = pb["person_id"].astype("int64")
    pb["comment"] = pb["comment"].fillna("").astype(str)
    box = {gid: d for gid, d in pb.groupby("game_id", sort=False)}
    # injury report: per game, rows sorted by publication time
    rep = {}
    if inj is not None and len(inj):
        ii = inj.dropna(subset=["game_id", "person_id"]).copy()
        ii["person_id"] = ii["person_id"].astype("int64")
        ii = ii.sort_values("report_utc")
        rep = {gid: d for gid, d in ii.groupby("game_id", sort=False)}
    lam_v = 0.5 ** (1.0 / half_life_games)
    lam_m = 0.5 ** (1.0 / minutes_half_life)
    players: dict[int, PlayerState] = defaultdict(PlayerState)
    team_players: dict = defaultdict(set)       # team -> players whose latest row is for that team
    team_idx: dict = defaultdict(int)          # number of games played by team so far
    team_hist: dict = defaultdict(list)         # realised lost per past game
    out = []
    for r in g.itertuples(index=False):
        rows = box.get(r.game_id)
        played = rows is not None and len(rows) > 0 and not math.isnan(r.home_pts if r.home_pts is not None else float("nan"))
        rec = {"game_id": r.game_id}
        rg = rep.get(r.game_id)
        for side, team in (("home", r.home), ("away", r.away)):
            ti = team_idx[team]
            cands = [(pid, players[pid]) for pid in team_players[team]
                     if ti - players[pid].last_team_idx <= window]
            vals = {}
            for pid, st in cands:
                v = (st.vn + prior_min * v0 / 36.0) / (st.vd + prior_min) * 36.0
                m = st.mn / st.md if st.md > 0 else 0.0
                vals[pid] = (max(v - v_rep, 0.0) * m / 36.0, v, m)
            # realised availability from this game's box score
            lost_real = np.nan
            if played:
                tr = rows[rows["team_id"] == team]
                avail = {}
                for x in tr.itertuples(index=False):
                    avail[x.person_id] = (x.min > 0) or not UNAVAILABLE.search(x.comment)
                if len(tr):
                    lost_real = sum(l for pid, (l, _, _) in vals.items() if not avail.get(pid, False))
            rec[f"{side}_lost_real"] = lost_real
            hist = team_hist[team]
            rec[f"{side}_lost_base"] = float(np.mean(hist[-base_n:])) if hist else 0.0
            rec[f"{side}_top_value"] = max((l for l, _, _ in vals.values()), default=0.0)
            rec[f"{side}_roster_value"] = sum(l for l, _, _ in vals.values())
            # point-in-time availability from the injury report at each decision offset
            for off in decision_offsets_min:
                key = f"{side}_lost_pit_{off}"
                if rg is not None:
                    t_dec = r.tip_utc - pd.Timedelta(minutes=off)
                    seen = rg[(rg["report_utc"] <= t_dec) & (rg["team_id"] == team)]
                    if len(seen):
                        last_t = seen["report_utc"].max()
                        latest = seen[seen["report_utc"] == last_t]
                        status = dict(zip(latest["person_id"], latest["current_status"]))
                        lost = 0.0
                        star_out = 0
                        top_pid = max(vals, key=lambda p: vals[p][0]) if vals else None
                        for pid, (l, _, _) in vals.items():
                            pp = P_PLAY.get(status.get(pid, "Available"), 1.0)
                            lost += (1 - pp) * l
                            if pid == top_pid and pp < 0.5:
                                star_out = 1
                        rec[key] = lost
                        rec[f"{side}_star_out_{off}"] = star_out
                        rec[f"{side}_report_age_min_{off}"] = (t_dec - last_t).total_seconds() / 60.0
                        rec[f"{side}_src_{off}"] = "report"
                        continue
                rec[key] = lost_real
                rec[f"{side}_star_out_{off}"] = np.nan
                rec[f"{side}_report_age_min_{off}"] = np.nan
                rec[f"{side}_src_{off}"] = "box" if played else "none"
        out.append(rec)
        # post-game updates
        if played:
            for side, team in (("home", r.home), ("away", r.away)):
                lr = rec[f"{side}_lost_real"]
                if lr == lr:
                    team_hist[team].append(lr)
                team_idx[team] += 1
            for x in rows.itertuples(index=False):
                st = players[x.person_id]
                if st.season is not None and st.season != r.season:
                    st.vn *= 0.6
                    st.vd *= 0.6
                    st.mn *= 0.3
                    st.md *= 0.3
                st.season = r.season
                if st.team != x.team_id:
                    if st.team is not None:
                        team_players[st.team].discard(x.person_id)
                    team_players[x.team_id].add(x.person_id)
                st.team = x.team_id
                st.last_team_idx = team_idx[x.team_id] - 1
                available = (x.min > 0) or not UNAVAILABLE.search(x.comment)
                if x.min > 0:
                    st.vn = lam_v * st.vn + x.gmsc
                    st.vd = lam_v * st.vd + x.min
                if available:
                    st.mn = lam_m * st.mn + x.min
                    st.md = lam_m * st.md + 1.0
    f = pd.DataFrame(out)
    for off in decision_offsets_min:
        f[f"lost_delta_{off}"] = (f[f"home_lost_pit_{off}"] - f["home_lost_base"]) - (
            f[f"away_lost_pit_{off}"] - f["away_lost_base"])
    f["lost_delta_real"] = (f["home_lost_real"] - f["home_lost_base"]) - (f["away_lost_real"] - f["away_lost_base"])
    return f
