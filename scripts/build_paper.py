"""Render the research paper from paper/paper_template.md.

Every number and table in the paper is pulled from outputs/ at build time, so the text can
never drift from the results. Placeholders:

  {{v:<league>/<file>.json:<key.path>|<fmt>}}            value from a JSON result file
  {{c:<league>/<file>.csv:<key col>=<value>:<col>|<fmt>}}  one cell of a CSV
  {{t:<league>/<file>.csv|<col>:<header>:<fmt>,...|<filter>}}  a Markdown table
  {{x:<name>|<fmt>}}                                        derived value computed below

fmt is a Python format spec, with two extras: "pct" (x100, 1 dp, %) and "pp" (x100, 2 dp).
Writes paper/paper.md (resolved) and paper/paper.html (standalone, figures inlined).
"""
from __future__ import annotations

import base64
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sq.config import OUT, ROOT  # noqa: E402

PAPER = ROOT / "paper"


def fmt(v, f):
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    if f == "pct":
        return f"{100 * float(v):.1f}%"
    if f == "pct0":
        return f"{100 * float(v):.0f}%"
    if f == "pp":
        return f"{100 * float(v):+.2f}"
    if f == "pts":                      # unsigned percentage points
        return f"{100 * float(v):.2f}"
    if f == "spct":                     # signed percent
        return f"{100 * float(v):+.1f}%"
    if f == "int":
        return f"{int(round(float(v))):,}"
    if f:
        return format(float(v), f)
    return str(v)


def jget(path, key):
    d = json.loads((OUT / path).read_text())
    for k in key.split("."):
        d = d[int(k)] if isinstance(d, list) else d[k]
    return d


def derived() -> dict:
    nba, nfl = OUT / "nba", OUT / "nfl"
    x = {}
    pk = pd.read_csv(nba / "market_kalshi_paired_vs_close.csv").set_index("model")
    x["nba_ens_gap"] = pk.loc["Ensemble", "mean_diff"]
    x["nba_ens_gap_lo"] = pk.loc["Ensemble", "ci_lo"]
    x["nba_ens_gap_hi"] = pk.loc["Ensemble", "ci_hi"]
    x["nba_elo_gap"] = pk.loc["Elo (538 params)", "mean_diff"]
    lr = pd.read_csv(nba / "longrun_paired_vs_elo538.csv").set_index("model")
    # reported as an improvement (positive = ensemble better than Elo)
    x["nba_ens_vs_elo"] = -lr.loc["Ensemble", "mean_diff"]
    x["nba_ens_vs_elo_lo"] = -lr.loc["Ensemble", "ci_hi"]
    x["nba_ens_vs_elo_hi"] = -lr.loc["Ensemble", "ci_lo"]
    x["nba_glicko_vs_elo_p"] = lr.loc["Glicko-2", "p_value"]
    x["nba_max_vs_elo_p"] = lr["p_value"].max()
    km = pd.read_csv(nba / "market_kalshi_metrics.csv").set_index("model")
    for k, lab in (("k_close", "Kalshi close"), ("k_open", "Kalshi open"), ("k_t24", "Kalshi T-24h"),
                   ("k_t6", "Kalshi T-6h"), ("ens", "Ensemble"), ("ens_e", "Ensemble (T-6h info)"),
                   ("elo", "Elo (538 params)")):
        x[f"nba_ll_{k}"] = km.loc[lab, "log_loss"]
    enc = pd.read_csv(nba / "market_kalshi_encompassing.csv").set_index("model")
    x["nba_enc_ens_b"] = enc.loc["Ensemble", "b_model"]
    x["nba_enc_ens_se"] = enc.loc["Ensemble", "se_model"]
    x["nba_enc_ens_p"] = enc.loc["Ensemble", "p_model"]
    x["nba_enc_min_p"] = enc["p_model"].min()
    pe = pd.read_csv(nba / "market_polymarket_encompassing.csv").set_index("model")
    x["nba_enc_pm_ens_b"] = pe.loc["Ensemble", "b_model"]
    x["nba_enc_pm_ens_p"] = pe.loc["Ensemble", "p_model"]
    nb = pd.read_csv(nfl / "book_ml_paired_vs_shin.csv").set_index("model")
    x["nfl_best_gap"] = nb.loc["Logistic", "mean_diff"]
    x["nfl_best_gap_lo"] = nb.loc["Logistic", "ci_lo"]
    x["nfl_best_gap_hi"] = nb.loc["Logistic", "ci_hi"]
    ne = pd.read_csv(nfl / "book_ml_encompassing.csv").set_index("model")
    x["nfl_enc_min_p"] = ne["p_model"].min()
    eff = pd.read_csv(nba / "efficiency_tests.csv")
    fam = eff[eff["in_family"]]
    x["nba_eff_n_tests"] = len(fam)
    x["nba_eff_min_pbh"] = fam["p_bh"].min()
    x["nba_eff_n_raw05"] = int((fam["p_value"] < 0.05).sum())
    eff2 = pd.read_csv(nfl / "efficiency_tests.csv")
    fam2 = eff2[eff2["in_family"]]
    x["nfl_eff_n_tests"] = len(fam2)
    x["nfl_eff_min_pbh"] = fam2["p_bh"].min()
    def row(test, term):
        r = eff[(eff["test"] == test) & (eff["term"] == term)]
        return r.iloc[0] if len(r) else None

    r = row("kalshi: price move on late injury news", "late_news")
    x["nba_reprice_kalshi"] = r["coef"] if r is not None else float("nan")
    for venue in ("kalshi", "polymarket"):
        r = row(f"{venue}: price move toward T-24h model gap", "gap_24h")
        x[f"drift_{venue}"], x[f"drift_{venue}_se"], x[f"drift_{venue}_r2"] = (
            (r["coef"], r["se"], r["r2"]) if r is not None else (float("nan"),) * 3)
    es = pd.read_csv(nba / "effect_sizes.csv").set_index("term")
    for t in ("b2b_diff", "home_altitude", "tz_diff", "lost_delta_5", "travel_diff", "rest_diff"):
        x[f"nba_eff_{t}"] = es.loc[t, "coef"]
        x[f"nba_eff_{t}_abs"] = abs(es.loc[t, "coef"])
        x[f"nba_eff_{t}_se"] = es.loc[t, "se"]
    x["nba_reprice_kalshi_abs"] = abs(x["nba_reprice_kalshi"])
    # NFL wind subgroup: two-sided normal-approximation test of over rate vs devigged price
    from scipy.stats import norm

    sj = json.loads((nfl / "summary.json").read_text())
    n_w, o_w, p_w = sj["total_n_wind15"], sj["total_over_rate_wind15"], sj["total_book_p_over_wind15"]
    z = (o_w - p_w) / (p_w * (1 - p_w) / n_w) ** 0.5
    x["nfl_wind_p"] = 2 * norm.sf(abs(z))
    bt = json.loads((nba / "backtests.json").read_text())
    for key, lab in (("kalshi|t6h|p_ensemble_early", "ens6"), ("kalshi|t6h|elo538_p_home", "elo6"),
                     ("kalshi|t24h|p_ensemble_d1", "ens24"), ("kalshi|t24h|p_ensemble_early", "leak24"),
                     ("kalshi|close|p_ensemble_close", "ensc"), ("kalshi|t6h|p_combined_early", "comb6"),
                     ("polymarket|t6h|p_ensemble_early", "pm6"), ("kalshi|t6h|p_kalman_plus_early", "kp6")):
        s = bt.get(key, {})
        for f in ("n_bets", "roi", "avg_clv", "avg_clv_after_cost", "max_drawdown", "exp_roi_at_close",
                  "p_roi_le_0", "total_return", "universe_games", "hit_rate"):
            x[f"bt_{lab}_{f}"] = s.get(f)
        ci = s.get("roi_ci") or [None, None]
        x[f"bt_{lab}_roi_lo"], x[f"bt_{lab}_roi_hi"] = ci
        cc = s.get("clv_ci") or [None, None]
        x[f"bt_{lab}_clv_lo"], x[f"bt_{lab}_clv_hi"] = cc
        if s.get("avg_clv") is not None and s.get("avg_clv_after_cost") is not None:
            x[f"bt_{lab}_avg_fee"] = s["avg_clv"] - s["avg_clv_after_cost"]
    nb2 = json.loads((nfl / "backtests.json").read_text())
    s = nb2.get("book|close|p_ensemble", {})
    x["bt_nflbook_n"], x["bt_nflbook_roi"], x["bt_nflbook_dd"] = s.get("n_bets"), s.get("roi"), s.get("max_drawdown")
    ci = s.get("roi_ci") or [None, None]
    x["bt_nflbook_lo"], x["bt_nflbook_hi"] = ci
    s = nb2.get("kalshi|t6h|p_ensemble", {})
    x["bt_nflk6_n"], x["bt_nflk6_roi"], x["bt_nflk6_clv"] = s.get("n_bets"), s.get("roi"), s.get("avg_clv")
    ci = s.get("roi_ci") or [None, None]
    x["bt_nflk6_lo"], x["bt_nflk6_hi"] = ci
    return x


def md_table(path, spec, filt):
    df = pd.read_csv(OUT / path)
    if filt:
        k, v = filt.split("=", 1)
        vals = v.split(";")
        df = df[df[k].astype(str).isin(vals)]
        if df[k].astype(str).is_unique:   # keep the order given in the filter
            order = {val: i for i, val in enumerate(vals)}
            df = df.assign(_o=df[k].astype(str).map(order)).sort_values("_o").drop(columns="_o")
    cols = [c.split(":") for c in spec.split(",")]
    head = "| " + " | ".join(h for _, h, *_ in cols) + " |"
    sep = "|" + "|".join("---" if i == 0 else "---:" for i in range(len(cols))) + "|"
    lines = [head, sep]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            name, f = c[0], (c[2] if len(c) > 2 else "")
            cells.append(fmt(r[name], f) if f else str(r[name]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def resolve(text: str) -> str:
    X = derived()

    def sub(m):
        kind, body = m.group(1), m.group(2)
        if kind == "v":
            ref, f = (body.split("|") + [""])[:2]
            path, key = ref.split(":", 1)
            return fmt(jget(path, key), f)
        if kind == "c":
            ref, f = (body.split("|") + [""])[:2]
            path, cond, col = ref.split(":", 2)
            k, v = cond.split("=", 1)
            df = pd.read_csv(OUT / path)
            return fmt(df.loc[df[k].astype(str) == v, col].iloc[0], f)
        if kind == "t":
            parts = body.split("|")
            return md_table(parts[0], parts[1], parts[2] if len(parts) > 2 else "")
        if kind == "x":
            name, f = (body.split("|") + [""])[:2]
            return fmt(X[name], f)
        return m.group(0)

    out = re.sub(r"\{\{([vctx]):(.+?)\}\}", sub, text)
    left = re.findall(r"\{\{.+?\}\}", out)
    if left:
        raise ValueError(f"unresolved placeholders: {left[:5]}")
    return out


CSS = """
:root{color-scheme:light;--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--rule:#e1e0d9;--accent:#2a78d6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--rule:#2c2c2a;--accent:#3987e5}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--rule:#2c2c2a;--accent:#3987e5}
body{background:var(--bg);color:var(--ink);font:16px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;margin:0}
main{max-width:860px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:28px;line-height:1.25;margin:0 0 8px}h2{font-size:21px;margin:40px 0 8px;border-top:1px solid var(--rule);padding-top:20px}
h3{font-size:17px;margin:28px 0 6px}p,li{color:var(--ink)}a{color:var(--accent)}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:12px 0 20px;display:block;overflow-x:auto}
th,td{padding:5px 9px;border-bottom:1px solid var(--rule);text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--ink2);font-weight:600}
img{max-width:100%;height:auto;border:1px solid var(--rule);border-radius:8px;background:#fcfcfb}
figure{margin:20px 0}figcaption{color:var(--ink2);font-size:13.5px;margin-top:6px}
code{font-size:13.5px}blockquote{border-left:3px solid var(--rule);margin:0;padding:2px 14px;color:var(--ink2)}
.meta{color:var(--ink2);font-size:14px}
"""


def to_html(md_text: str) -> str:
    import markdown

    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "toc", "attr_list"])

    def inline(m):
        src = m.group(1)
        p = (PAPER / src).resolve()
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode()
            return f'src="data:image/png;base64,{b64}"'
        return m.group(0)

    body = re.sub(r'src="([^"]+\.png)"', inline, body)
    title = re.search(r"<h1[^>]*>(.*?)</h1>", body)
    t = re.sub("<[^>]+>", "", title.group(1)) if title else "Paper"
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width, initial-scale=1'><title>{t}</title><style>{CSS}</style></head>"
            f"<body><main>{body}</main></body></html>")


REG = OUT / "nba" / "registered"     # registered props record (discovery + replication), frozen


def lines_props_section() -> str:
    p = OUT / "nba" / "lines_props.json"
    if not p.exists():
        return "_Not run: `scripts/run_research.py` writes `outputs/nba/lines_props.json`._"
    r = json.loads(p.read_text())
    st, default_props = r["spreads_totals"], r.get("props", {})
    # the props narrative starts from the registered discovery record (box-score scoring, as registered);
    # the default pipeline's settlement-scored results enter in the correction below
    preg = REG / "lines_props.json"
    pr = json.loads(preg.read_text())["props"] if preg.exists() else {}
    sp, to = st["spread"], st["total"]
    out = [
        f"**Spreads and totals (Polymarket main lines).** Outcomes were reconstructed from box scores and checked "
        f"against each market's own settlement (agreement {100 * st['spread_settlement_agreement']:.0f}% for spreads, "
        f"{100 * st['total_settlement_agreement']:.0f}% for totals, after parsing the line from the market question: "
        f"the API's line field sometimes drops the minus sign). On {sp['n']:,} NBA games with a closing Polymarket "
        f"spread, home teams covered {100 * sp['cover_rate_home']:.1f}% against an average closing price of "
        f"{100 * sp['mkt_mean_p']:.1f}%. On {to['n']:,} totals the over hit {100 * to['over_rate']:.1f}% against "
        f"{100 * to['mkt_mean_p']:.1f}%. Prices sit near 50% by design, so log loss and Brier are the informative "
        f"scores here.",
        "",
        md_table("nba/nba_spread_metrics.csv", "model:Spread forecast,n:Games:int,log_loss:Log loss:.4f,brier:Brier:.4f", ""),
        "",
        md_table("nba/nba_total_metrics.csv", "model:Total forecast,n:Games:int,log_loss:Log loss:.4f,brier:Brier:.4f", ""),
        "",
        f"Encompassing: the state-space model's weight next to the spread price is {sp['encompassing']['b_model']:.2f} "
        f"(p = {sp['encompassing']['p_model']:.2f}) and next to the total price {to['encompassing']['b_model']:.2f} "
        f"(p = {to['encompassing']['p_model']:.2f}). The model's mean total error is "
        f"{to['model_bias_pts']:+.1f} points on these games.",
        "",
    ]
    if pr.get("available"):
        cm, enc, pt = pr["calibration_market"], pr["encompassing"], pr["paired_model_minus_market"]
        out += [
            f"**Player points props (Kalshi, random 15% sample of games, markets with at least 500 contracts traded).** "
            f"{pr['n_scored']:,} closing prop prices across "
            f"{pr['games']:,} games and {pr['players']:,} players (player match rate {100 * pr['matched_players_share']:.0f}%). "
            f"Mean closing spread {100 * pr['mean_spread']:.1f} points; {100 * pr['yes_rate']:.1f}% of props resolved Yes "
            f"against a mean price of {100 * pr['mean_price']:.1f}%; {100 * pr['dnp_share']:.1f}% of scored props were on "
            f"players who did not play, which the code scored as No. Kalshi does not settle them that way; see the "
            f"correction at the end of this section.",
            "",
            md_table("nba/registered/props_metrics.csv", "model:Prop forecast,n:Props:int,log_loss:Log loss:.4f,brier:Brier:.4f,"
                     "cal_slope:Calibration slope:.2f,cal_intercept:Intercept:.2f", ""),
            "",
            f"The prop market's calibration slope is {cm['slope']:.2f} (SE {cm['slope_se']:.2f}), so no "
            f"favourite-longshot tilt, but Yes outcomes run {100 * pr['yes_at_ask']['mean_gap_vs_mid']:.1f} points "
            f"above the mid-implied probability. That is about the average half-spread "
            f"({100 * pr['yes_at_ask']['mean_half_spread']:.1f} points): buying every Yes at the closing ask plus the "
            f"taker fee returns {100 * pr['yes_at_ask']['roi']:+.1f}% (95% CI {100 * pr['yes_at_ask']['roi_ci'][0]:+.1f}% "
            f"to {100 * pr['yes_at_ask']['roi_ci'][1]:+.1f}%, resampling games). The mid is a slightly low estimate of "
            f"the Yes probability on these books, and the executable price is fair.",
            "",
            f"The player model (negative binomial, dispersion {pr['dispersion']:.2f}, availability from the injury "
            f"report) is {pt['mean_diff']:+.4f} log loss versus the market (95% CI {pt['ci_lo']:+.4f} to "
            f"{pt['ci_hi']:+.4f}, clustered by game). Its encompassing weight next to the prop price is "
            f"{enc['b_model']:.2f} (SE {enc['se_model']:.2f}, p = {enc['p_model']:.3f}). This looked like the one place in "
            f"the study where a model carries information the closing price lacks. It was a single test on "
            f"{pr['games']:,} games that would not survive the multiple-testing correction applied to Section 6, so it "
            f"was put to a pre-registered replication instead of being reported as a finding.",
        ]
    rp = REG / "props_replication.json"
    rx = OUT / "nba" / "props_rep_exploratory.json"
    ex = json.loads(rx.read_text()) if rx.exists() else None
    if rp.exists():
        r = json.loads(rp.read_text())
        e, v = r["encompassing"], r["verdict"]
        h3, h3b, ya = r["strategy_h3_combined"], r["strategy_h3"], r["yes_at_ask"]

        def roi(s, n=None):
            if "roi" not in s:
                return "no bets"
            n = s.get("n_bets", n)
            return (f"{100 * s['roi']:+.1f}% on {n:,} bets (95% CI {100 * s['roi_ci'][0]:+.1f}% to "
                    f"{100 * s['roi_ci'][1]:+.1f}%)")

        timing = ""
        if ex and ex.get("registration_timing", {}).get("first_props_request_after"):
            t = ex["registration_timing"]
            timing = (f" The registration file was saved at {t['registration_file_created']}; in the Kalshi fetch "
                      f"manifest the last request for this series before then was the discovery download "
                      f"({t['last_props_request_before']}) and the first after it came "
                      f"{t['seconds_after_registration']:.0f} seconds later.")
        h1_word = ("replicated" if v["H1_model_adds_information"] else "not replicated") + (
            " as registered" if ex else "")
        out += [
            "",
            "**Pre-registered replication.** The follow-up was registered before any new data was downloaded "
            "(`paper/preregistration_props.md`: hypotheses, filters, frozen code hashes and frozen combination "
            f"weights).{timing} On held-out games ({r['n_scored']:,} closing prices, {r['games']:,} games, "
            f"{r['players']:,} players, {r['dates'][0]} to {r['dates'][1]}, the same months as the discovery sample but "
            f"no overlap with its games):",
            "",
            md_table("nba/registered/props_rep_metrics.csv", "model:Prop forecast,n:Props:int,log_loss:Log loss:.4f,"
                     "brier:Brier:.4f,cal_slope:Calibration slope:.2f", ""),
            "",
            f"* H1, information: the player model's encompassing weight is {e['b_model']:.3f} (game-clustered SE "
            f"{e['se_model']:.3f}, one-sided p = {e['p_model_one_sided']:.4f}): **{h1_word}**"
            + (" (but see the correction below)." if ex else "."),
            f"* H2, buying every Yes at the ask plus fee: {roi(ya, r['n_scored'])}; "
            f"{'not profitable, as predicted' if v['H2_yes_at_ask_not_profitable'] else 'profitable, contrary to prediction'}.",
            f"* H3, frozen model-plus-market combination, flat stakes: {roi(h3)}; "
            f"{'profitable' if v['H3_combined_rule_profitable'] else 'not profitable'}.",
            f"* H3b, player model alone: {roi(h3b)}; "
            f"{'profitable' if v['H3b_model_only_rule_profitable'] else 'not profitable'}.",
        ]
        if ex:
            tr = ex["trading_rules"]
            out += [
                "",
                f"The combination rule bought Yes on {100 * tr['h3_share_yes']:.0f}% of its bets. Its frozen intercept "
                f"carries the discovery sample's Yes premium ({100 * tr['yes_premium_vs_mid_discovery']:.1f} points "
                f"above the mid), but in the replication Yes outcomes ran only "
                f"{100 * tr['yes_premium_vs_mid_replication']:.1f} points above the mid, less than the "
                f"{100 * tr['half_spread_replication']:.1f}-point half-spread paid at the ask.",
            ]
    if ex:
        out += ["", props_correction(ex, default_props)]
    return "\n".join(out)


def default_pipeline_sentence(ex: dict, d: dict) -> str:
    """One sentence on the default (settlement-scored) props pipeline, checked against the post-hoc column."""
    if not d.get("available") or d.get("scoring") != "settlement":
        return ""
    e = ex["settled_discovery"]["encompassing_binary_settled"]["registered_model"]
    same = (d["n_scored"] == e["n"] and abs(d["encompassing"]["b_model"] - e["b_model"]) < 1e-9
            and abs(d["encompassing"]["se_model"] - e["se_game"]) < 1e-9)
    agree = f"{100 * d['box_score_settlement_agreement']:.0f}%"
    if same:
        return (f" The props pipeline (`sq/research/nba_lines.py`) now scores outcomes from settlement by default and "
                f"reproduces the discovery \"as settled\" column exactly; the box-score outcome is kept as a check and "
                f"agrees with {agree} of binary settlements. The registered code is preserved byte-for-byte as "
                f"`sq/research/nba_lines_registered.py`, with its outputs in `outputs/nba/registered/`.")
    return (f" The props pipeline (`sq/research/nba_lines.py`) now scores outcomes from settlement by default; on the "
            f"discovery sample it gives a model weight of {d['encompassing']['b_model']:.3f} "
            f"(SE {d['encompassing']['se_model']:.3f}) on {d['n_scored']:,} props, which does not match the post-hoc "
            f"column above; see `outputs/nba/lines_props.json`.")


def props_correction(ex: dict, default_props: dict | None = None) -> str:
    """Post-hoc correction of the props outcome definition (sq/research/props_exploratory.py)."""
    sd, sr = ex["settled_discovery"], ex["settled_replication"]

    def ret(s):
        if "roi" not in s:
            return "no bets"
        return f"{100 * s['roi']:+.1f}% ({100 * s['roi_ci'][0]:+.1f} to {100 * s['roi_ci'][1]:+.1f})"

    def col(d, settled):
        if settled:
            e = d["encompassing_binary_settled"]["registered_model"]
            return {"n": e["n"], "b": e["b_model"], "se": e["se_game"], "p": e["p_one_sided_game"],
                    "comb": d["paired_combined_minus_market_binary_settled"]["mean_diff"],
                    "ya": d["yes_at_ask_settled"], "h3": d["h3_combined_settled"], "h3b": d["h3b_model_settled"]}
        b = d["box_score"]
        return {"n": d["n"], "b": b["encompassing"]["b_model"], "se": b["encompassing"]["se_model"],
                "p": b["encompassing"]["p_model_one_sided"], "comb": b["paired_combined_minus_market"]["mean_diff"],
                "ya": b["yes_at_ask"], "h3": b["h3_combined"], "h3b": b["h3b_model"]}

    cols = [col(sd, False), col(sd, True), col(sr, False), col(sr, True)]
    rows = [("Props scored", lambda c, i: f"{c['n']:,}"),
            ("Player-model weight (SE)", lambda c, i: f"{c['b']:.3f} ({c['se']:.3f})"),
            ("One-sided p, game-clustered", lambda c, i: f"{c['p']:.4f}"),
            ("Frozen combination minus market, log loss",
             lambda c, i: f"{c['comb']:+.4f}" + (" (in sample)" if i < 2 else "")),
            ("Yes at ask + fee: return (95% CI)", lambda c, i: ret(c["ya"])),
            ("H3 combination rule: return (95% CI)", lambda c, i: ret(c["h3"])),
            ("H3b model-only rule: return (95% CI)", lambda c, i: ret(c["h3b"]))]
    table = ["| | Discovery, DNP scored No | Discovery, as settled | Replication, DNP scored No | "
             "Replication, as settled |", "|---|---:|---:|---:|---:|"]
    table += ["| " + lab + " | " + " | ".join(f(c, i) for i, c in enumerate(cols)) + " |" for lab, f in rows]
    er = sr["encompassing_binary_settled"]["registered_model"]
    ed = sd["encompassing_binary_settled"]["registered_model"]
    ps = [c["p_one_sided"] for c in ex["clustering_replication"] if c["clustering"] != "Game (registered)"]
    return "\n".join([
        "**A scoring error behind both results (found after the replication; not pre-registered).** The registered "
        "code scores every prop from the box score, so a player who does not play counts as a No. Kalshi does not "
        "settle these markets No: the contract terms say a player who is active but never takes the court settles at "
        "the last fair market price before the game, and in the data every prop on a player who did not play, "
        "including players listed Out, was settled at such a price (result \"scalar\"; mean settlement value "
        f"{sr['scalar_settlement_value_mean']:.2f} in the replication sample). Those props carry essentially the "
        f"whole signal. The replication sample has {sr['n_scalar_dnp']} of them ({sd['n_scalar_dnp']} in discovery), "
        f"{sr['n_scalar_listed_out']} on players the injury report had already listed Out, which the model, reading "
        f"the report, prices at its 0.5% floor. Across the scalar-settled props the closing mid averages "
        f"{sr['scalar_mid_mean']:.2f} (mean spread {100 * sr['scalar_spread_mean']:.0f} points): the market expects a "
        "fair-price settlement, not a zero. Below, the same analyses are scored the way the markets settled: "
        "scalar-settled props are left out of the 0/1 regression and paid at their settlement value in the trading "
        "rules"
        + (f" ({sr['n_scalar'] - sr['n_scalar_dnp']} further scalar-settled props, on players who did play, are "
           "treated the same way)" if sr["n_scalar"] > sr["n_scalar_dnp"] else "")
        + "; the model, filters, frozen coefficients and costs are unchanged (`sq/research/props_exploratory.py`).",
        "",
        *table,
        "",
        f"Scored as settled, the player model adds nothing to the closing price in either sample: its weight is "
        f"{er['b_model']:.3f} (SE {er['se_game']:.3f}) on the held-out games and {ed['b_model']:.2f} "
        f"(SE {ed['se_game']:.2f}) in discovery, the frozen combination scores the same as the market, and every "
        f"trading rule loses money on the held-out games. The registered H1 result is not a standard-error artifact: clustering by player, "
        f"by date or two ways gives one-sided p between {min(ps):.4f} and {max(ps):.4f}. The outcome was "
        f"mismeasured, identically in both samples, because the replication reused the frozen discovery code. "
        f"Pre-registration guards against selective analysis, not against an error shared by discovery and "
        f"replication. Spread and total outcomes were checked against each market's own settlement (above); the props "
        f"outcome should have been too, and the settlement-consistent numbers are the ones to rely on."
        + game_settlement_sentence(ex.get("game_market_settlement") or {})
        + default_pipeline_sentence(ex, default_props or {}),
    ])


def game_settlement_sentence(g: dict) -> str:
    nb, nf = g.get("nba"), g.get("nfl")
    if not nb or not nf:
        return ""
    if nb["matched_to_played_games"] or nf["matched_non_ties"]:
        return (f" Some scalar-settled game-winner markets match played games that are scored 0/1 (NBA "
                f"{nb['matched_to_played_games']}, NFL non-ties {nf['matched_non_ties']}); see "
                f"`outputs/nba/props_rep_exploratory.json`.")
    return (f" The same audit of the game-winner markets finds nothing to correct: none of the "
            f"{nb['scalar_settled_events']} NBA games Kalshi settled at a scalar price matches a game played on its "
            f"date, and the {nf['matched_ties']} NFL games that do match are ties (settled at 0.50), which are already "
            f"excluded from scoring.")


def props_abstract() -> str:
    rp, rx = REG / "props_replication.json", OUT / "nba" / "props_rep_exploratory.json"
    if not (rp.exists() and rx.exists()):
        return ""
    r, ex = json.loads(rp.read_text()), json.loads(rx.read_text())
    er = ex["settled_replication"]["encompassing_binary_settled"]["registered_model"]
    return (f" The one apparent exception, a player-points model that seemed to add information to Kalshi prop "
            f"prices, was put to a pre-registered replication on {r['games']:,} held-out games. It passed as "
            f"registered (one-sided p = {r['encompassing']['p_model_one_sided']:.4f}), but a post-hoc check traced the "
            f"signal in both samples to a scoring error: props on players who sat out were scored as losses, while "
            f"Kalshi settles them at a pre-game fair price. Scored as they settle, the model's weight is "
            f"{er['b_model']:.2f} (SE {er['se_game']:.2f}).")


if __name__ == "__main__":
    tpl = (PAPER / "paper_template.md").read_text(encoding="utf-8")
    tpl = tpl.replace("__LINES_PROPS__", lines_props_section())
    tpl = tpl.replace("__PROPS_ABSTRACT__", props_abstract())
    md = resolve(tpl)
    (PAPER / "paper.md").write_text(md, encoding="utf-8")
    (PAPER / "paper.html").write_text(to_html(md), encoding="utf-8")
    print("wrote", PAPER / "paper.md", "and paper.html")
