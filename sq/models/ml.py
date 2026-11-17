"""Walk-forward statistical-learning models: logistic regression and gradient boosting.

Every prediction for season s comes from a model fitted on seasons < s only, using
features that are themselves pre-game values (online model outputs, schedule,
point-in-time availability). Hyperparameters are chosen on the last training
season (an inner validation split), never on the test season.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..eval.metrics import log_loss


def _xy(df, features, target):
    X = df[features].astype(float).to_numpy()
    y = df[target].astype(float).to_numpy()
    return X, y


def walk_forward_logistic(df: pd.DataFrame, features: list[str], test_seasons, target="y", first_season=None,
                          Cs=(0.003, 0.01, 0.03, 0.1, 0.3, 1.0), name="logit") -> pd.DataFrame:
    out = []
    d = df.dropna(subset=features + [target]) if target in df else df
    for s in test_seasons:
        tr = d[(d["season"] < s) & ((d["season"] >= first_season) if first_season else True)]
        te = df[df["season"] == s].dropna(subset=features)
        if len(tr) < 200 or te.empty:
            continue
        val_season = tr["season"].max()
        inner, val = tr[tr["season"] < val_season], tr[tr["season"] == val_season]
        best_c, best = Cs[0], np.inf
        for c in Cs:
            m = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
            m.fit(*_xy(inner, features, target))
            ll = log_loss(m.predict_proba(val[features].astype(float).to_numpy())[:, 1], val[target])
            if ll < best:
                best, best_c = ll, c
        m = make_pipeline(StandardScaler(), LogisticRegression(C=best_c, max_iter=2000))
        m.fit(*_xy(tr, features, target))
        p = m.predict_proba(te[features].astype(float).to_numpy())[:, 1]
        coefs = dict(zip(features, m[-1].coef_[0] / m[0].scale_))
        out.append(pd.DataFrame({"game_id": te["game_id"].to_numpy(), f"p_{name}": p, "season": s,
                                 f"{name}_C": best_c}).assign(**{f"coef_{k}": v for k, v in coefs.items()}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def walk_forward_gbm(df: pd.DataFrame, features: list[str], test_seasons, target="y", first_season=None,
                     name="gbm", params=None, monotone=None) -> pd.DataFrame:
    import lightgbm as lgb

    base = dict(n_estimators=2000, learning_rate=0.02, num_leaves=15, min_child_samples=200, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1)
    if params:
        base.update(params)
    if monotone:
        base["monotone_constraints"] = [monotone.get(f, 0) for f in features]
    out = []
    d = df.dropna(subset=[target])
    for s in test_seasons:
        tr = d[(d["season"] < s) & ((d["season"] >= first_season) if first_season else True)]
        te = df[df["season"] == s]
        if len(tr) < 500 or te.empty:
            continue
        val_season = tr["season"].max()
        inner, val = tr[tr["season"] < val_season], tr[tr["season"] == val_season]
        m = lgb.LGBMClassifier(**base)
        m.fit(inner[features].astype(float), inner[target].astype(int),
              eval_set=[(val[features].astype(float), val[target].astype(int))], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(100, verbose=False)])
        n_best = max(int(m.best_iteration_ or 100), 50)
        m2 = lgb.LGBMClassifier(**{**base, "n_estimators": int(n_best * 1.1)})
        m2.fit(tr[features].astype(float), tr[target].astype(int))
        p = m2.predict_proba(te[features].astype(float))[:, 1]
        out.append(pd.DataFrame({"game_id": te["game_id"].to_numpy(), f"p_{name}": p, "season": s,
                                 f"{name}_trees": int(n_best * 1.1)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def walk_forward_margin_adjust(df: pd.DataFrame, mu_col: str, sd_col: str, adj_cols: list[str], test_seasons,
                               margin_col="margin", first_season=None, name="kalman_plus") -> pd.DataFrame:
    """Add linear adjustments (e.g. player availability, rest) to a Gaussian margin forecast.

    residual = margin - mu is regressed on ``adj_cols`` with training seasons only; the
    adjusted mean is mu + X b and the predictive SD is the training residual SD.
    """
    from scipy.special import ndtr

    out = []
    d = df.dropna(subset=[margin_col, mu_col] + adj_cols)
    for s in test_seasons:
        tr = d[(d["season"] < s) & ((d["season"] >= first_season) if first_season else True)]
        te = df[df["season"] == s].dropna(subset=[mu_col] + adj_cols)
        if len(tr) < 200 or te.empty:
            continue
        X = np.column_stack([tr[adj_cols].to_numpy(float)])
        r = (tr[margin_col] - tr[mu_col]).to_numpy(float)
        b, *_ = np.linalg.lstsq(X, r, rcond=None)
        resid = r - X @ b
        # inflate the model's own SD by the share of variance the adjustment explains (keeps calibration)
        scale = np.sqrt(np.mean(resid ** 2) / np.mean(r ** 2))
        mu = te[mu_col].to_numpy(float) + te[adj_cols].to_numpy(float) @ b
        sd = te[sd_col].to_numpy(float) * scale
        out.append(pd.DataFrame({"game_id": te["game_id"].to_numpy(), f"p_{name}": ndtr(mu / sd),
                                 f"mu_{name}": mu, f"sd_{name}": sd, "season": s}).assign(
            **{f"b_{c}": v for c, v in zip(adj_cols, b)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
