"""
nested_cv_feature_selection.py  —  ENGINE
=========================================
Leakage-free re-evaluation of the RERTA modelling pipeline. Feature selection
(or PCA) and model fitting are performed INSIDE every cross-validation fold, so
the held-out rows never influence which features/components are used. This keeps
the original dimensionality-reduction rationale intact (selection/PCA still
happen every fold) while removing the optimism of choosing them on all rows.

Supports the three analysis options from main.py:
  * "vi"       : VI + DEM statistics, dimensionality reduced by feature SELECTION
  * "bands"    : 7 raw ortho bands, dimensionality reduced by PCA (top components)
  * "bands+vi" : both, dimensionality reduced by feature SELECTION

and the estimators used in the report:
  * OLS        (linear baseline; LinearRegression == statsmodels OLS for prediction)
  * RF_tuned   (the project's own tune_random_forest)
  * RF_plain   (untuned RF, for fast sanity passes)

For each (option, target, estimator) it reports:
  LEAKY  OOF R2  : transform fit on ALL rows once, then CV   (current main.py behaviour)
  NESTED OOF R2  : transform re-fit on training rows every fold (honest)
  baseline   R2  : pooled out-of-fold R2 of a mean predictor (reference ~ 0)
  optimism       : LEAKY - NESTED

SAFETY: reads Data/*.gpkg READ-ONLY; the engine itself writes NOTHING.
Does not import main.py (that would execute its pipeline); the small config below
mirrors main.py and should be kept in sync.
"""

import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, LeaveOneOut
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rerta import statistical_modelling as sm  # safe: no top-level side effects

# ---------------------------------------------------------------------------
# CONFIG — mirrors scripts/run_analysis.py (vi / bands / bands+vi). Keep in sync.
# Post-refactor layout: package under rerta/, zonal-stat GeoPackages in data/processed/.
# ---------------------------------------------------------------------------
DATA = os.path.join(_REPO_ROOT, "data", "processed")

VI_SOURCES = [
    ("DEM",    os.path.join(DATA, "Palapa July2025 DEM statistics.gpkg")),
    ("GLI",    os.path.join(DATA, "Palapa July2025 GLI statistics.gpkg")),
    ("Clre",   os.path.join(DATA, "Palapa July2025 Clre statistics.gpkg")),
    ("ReNDVI", os.path.join(DATA, "Palapa July2025 ReNDVI statistics.gpkg")),
    ("GNDVI",  os.path.join(DATA, "Palapa July2025 GNDVI statistics.gpkg")),
    ("NDVI",   os.path.join(DATA, "Palapa July2025 NDVI statistics.gpkg")),
]
BAND_SOURCES = [
    (f"band{i}", os.path.join(DATA, f"Palapa July2025 ortho statistics_band{i}.gpkg"))
    for i in range(1, 8)
]
RESULT_GPKG = os.path.join(DATA, "Palapa_veg_plots_result.gpkg")

# name -> (sources, strategy) ; strategy in {"select", "pca"}
OPTIONS = {
    "vi":       (VI_SOURCES,                "select"),
    "bands":    (BAND_SOURCES,              "pca"),
    "bands+vi": (VI_SOURCES + BAND_SOURCES, "select"),
}

FEATURE_KEYS = ["DEM", "GLI", "Clre", "ReNDVI", "GNDVI", "NDVI", "band"]
TRANSFORMATIONS = {
    "proportion":      sm.arcsinc_sqrt_transform,
    "canopy_openness": sm.arcsinc_sqrt_transform,
    "abundance":       sm.log_transform,
    "richness":        sm.log_transform,
}
_EXCLUDE_TARGETS = {"geometry", "point.label", "treatment"}

RANDOM_STATE = 42
PCA_MAX_COMPONENTS = 4          # mirrors main.py: min(4, available)
RF_TUNE_N_ITER = 10             # main.py uses 20; reduced to keep nested CV tractable


# ---------------------------------------------------------------------------
# DATA RECONSTRUCTION (read-only; mirrors main.py minimally)
# ---------------------------------------------------------------------------
def reconstruct_dataset(sources):
    """Build a merged, snake-cased, transformed DataFrame for the given sources.

    Each statistics GeoPackage already carries the field target columns (they
    were merged in during zonal_statistics), so the first source supplies the
    targets. Returns (df, targets, features). Read-only.
    """
    import geopandas as gpd
    raw_targets = [c for c in gpd.read_file(RESULT_GPKG).columns
                   if c not in _EXCLUDE_TARGETS]

    merged = sm.load_data(sources, raw_targets, filter=None)

    meaningful = set(raw_targets) | {
        c for c in merged.columns if any(k in c for k in FEATURE_KEYS)
    }
    for col in list(merged.columns):
        if col in meaningful:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
        snake = col.replace(" ", "_").replace(".", "_")
        if snake != col:
            merged = merged.rename(columns={col: snake})
            col = snake
        for key, fn in TRANSFORMATIONS.items():
            if key in col:
                merged[col] = fn(merged[col])

    targets = [c.replace(" ", "_").replace(".", "_") for c in raw_targets]
    features = [c for c in merged.columns if any(k in c for k in FEATURE_KEYS)]
    return merged, targets, features


# ---------------------------------------------------------------------------
# ESTIMATOR FACTORIES
# ---------------------------------------------------------------------------
def ols_factory(x_train, y_train):
    """OLS baseline. LinearRegression gives identical point predictions to the
    project's statsmodels OLS (standardisation does not change linear fits)."""
    return LinearRegression().fit(np.asarray(x_train), np.asarray(y_train))


def rf_tuned_factory(x_train, y_train):
    """Tuned RF via the project's own tune_random_forest (RandomizedSearchCV)."""
    model, _ = sm.tune_random_forest(
        np.asarray(x_train), np.asarray(y_train),
        random_state=RANDOM_STATE, n_iter=RF_TUNE_N_ITER,
    )
    return model


def rf_plain_factory(x_train, y_train):
    """Untuned RF for a fast sanity pass."""
    return RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE)\
        .fit(np.asarray(x_train), np.asarray(y_train))


FACTORIES = {"OLS": ols_factory, "RF_tuned": rf_tuned_factory, "RF_plain": rf_plain_factory}


# ---------------------------------------------------------------------------
# FOLD TRANSFORMS  (the dimensionality reduction — fit per fold when nested)
# ---------------------------------------------------------------------------
def _select_features(frame, target, all_features):
    """Project selection pipeline with a graceful fallback on tiny/singular folds."""
    try:
        feats = sm.smart_feature_selection_pipeline(frame, target, all_features)
        if feats:
            return feats
    except Exception as exc:  # noqa: BLE001 - tiny folds can break VIF
        print(f"      [selection fell back: {type(exc).__name__}]")
    corrs = []
    for f in all_features:
        try:
            c = abs(frame[target].corr(frame[f], method="spearman"))
            if pd.notna(c):
                corrs.append((f, c))
        except Exception:  # noqa: BLE001
            continue
    corrs.sort(key=lambda t: t[1], reverse=True)
    return [f for f, _ in corrs[:3]] or all_features[:3]


def transform_select(train, test, target, all_features, fitted=None):
    """SELECT strategy. If `fitted` (a feature list) is given, reuse it (leaky);
    otherwise select on `train` only (nested)."""
    feats = fitted if fitted is not None else _select_features(train, target, all_features)
    return train[feats].to_numpy(), test[feats].to_numpy(), feats


def transform_pca(train, test, target, all_features, fitted=None):
    """PCA strategy. If `fitted` (scaler, pca) is given, reuse it (leaky);
    otherwise fit scaler+PCA on `train` only (nested)."""
    if fitted is not None:
        scaler, pca = fitted
    else:
        Xtr = train[all_features].to_numpy()
        scaler = StandardScaler().fit(Xtr)
        n_comp = min(PCA_MAX_COMPONENTS, Xtr.shape[1], max(1, len(train) - 1))
        pca = PCA(n_components=n_comp, random_state=RANDOM_STATE).fit(scaler.transform(Xtr))
    Xtr = pca.transform(scaler.transform(train[all_features].to_numpy()))
    Xte = pca.transform(scaler.transform(test[all_features].to_numpy()))
    return Xtr, Xte, (scaler, pca)


def _fit_full_transform(data, target, all_features, strategy):
    """Fit the transform once on ALL rows (the leaky reference)."""
    if strategy == "pca":
        X = data[all_features].to_numpy()
        scaler = StandardScaler().fit(X)
        n_comp = min(PCA_MAX_COMPONENTS, X.shape[1], max(1, len(data) - 1))
        pca = PCA(n_components=n_comp, random_state=RANDOM_STATE).fit(scaler.transform(X))
        return (scaler, pca)
    return _select_features(data, target, all_features)  # feature list


# ---------------------------------------------------------------------------
# CROSS-VALIDATION EVALUATORS
# ---------------------------------------------------------------------------
def make_cv(scheme="rkf", splits=5, repeats=3, seed=RANDOM_STATE):
    if scheme == "loo":
        return LeaveOneOut()
    return RepeatedKFold(n_splits=splits, n_repeats=repeats, random_state=seed)


def _transform_fn(strategy):
    return transform_pca if strategy == "pca" else transform_select


def evaluate(df, target, all_features, estimator_factory, strategy, cv, leaky):
    """Pooled out-of-fold R2. `leaky=True` fits the transform on all rows once;
    `leaky=False` re-fits it inside each training fold."""
    data = df[[target] + all_features].dropna().reset_index(drop=True)
    if len(data) < 10:
        return None
    y = data[target].to_numpy()
    tfn = _transform_fn(strategy)
    fitted_full = _fit_full_transform(data, target, all_features, strategy) if leaky else None

    oof_sum = np.zeros(len(data))
    oof_cnt = np.zeros(len(data))
    dims, chosen = [], Counter()
    for tr_idx, te_idx in cv.split(data):
        train, test = data.iloc[tr_idx], data.iloc[te_idx]
        Xtr, Xte, info = tfn(train, test, target, all_features, fitted=fitted_full)
        model = estimator_factory(Xtr, train[target].to_numpy())
        pred = model.predict(Xte)
        oof_sum[te_idx] += pred
        oof_cnt[te_idx] += 1
        if strategy == "select":
            dims.append(len(info)); chosen.update(info)
        else:
            dims.append(info[1].n_components_)

    mask = oof_cnt > 0
    oof_pred = oof_sum[mask] / oof_cnt[mask]
    return {
        "oof_r2": r2_score(y[mask], oof_pred),
        "n": len(data),
        "mean_dim": float(np.mean(dims)) if dims else float("nan"),
        "stability": chosen,
    }


def mean_baseline_r2(df, target, all_features, cv):
    data = df[[target] + all_features].dropna().reset_index(drop=True)
    if len(data) < 10:
        return float("nan")
    y = data[target].to_numpy()
    oof = np.full(len(data), np.nan)
    for tr_idx, te_idx in cv.split(data):
        oof[te_idx] = np.mean(y[tr_idx])
    return r2_score(y, np.where(np.isnan(oof), np.mean(y), oof))


# ---------------------------------------------------------------------------
# GRID RUNNER
# ---------------------------------------------------------------------------
def run_grid(option_names, estimator_names, cv=None, verbose=True):
    """Run the full {option x target x estimator} grid. Returns a list of dict
    rows. Pure computation — writes nothing."""
    if cv is None:
        cv = make_cv()
    import contextlib, io
    rows = []
    for opt in option_names:
        sources, strategy = OPTIONS[opt]
        with contextlib.redirect_stdout(io.StringIO()):       # silence framework prints
            df, targets, features = reconstruct_dataset(sources)
        if verbose:
            print(f"\n[option={opt}] {len(df)} plots, {len(features)} features, strategy={strategy}")
        for target in targets:
            if target not in df.columns or df[target].notna().sum() < 10:
                continue
            base = mean_baseline_r2(df, target, features, cv)
            for est in estimator_names:
                factory = FACTORIES[est]
                with contextlib.redirect_stdout(io.StringIO()):
                    leaky = evaluate(df, target, features, factory, strategy, cv, leaky=True)
                    nested = evaluate(df, target, features, factory, strategy, cv, leaky=False)
                if leaky is None or nested is None:
                    continue
                row = {
                    "option": opt, "target": target, "estimator": est,
                    "n": nested["n"], "leaky_r2": leaky["oof_r2"],
                    "nested_r2": nested["oof_r2"], "baseline_r2": base,
                    "optimism": leaky["oof_r2"] - nested["oof_r2"],
                    "mean_dim": nested["mean_dim"],
                    "top_feats": "; ".join(f"{f}({c})" for f, c in nested["stability"].most_common(4)),
                }
                rows.append(row)
                if verbose:
                    print(f"  {target[:34]:34s} {est:8s} "
                          f"leaky={row['leaky_r2']:+.3f}  nested={row['nested_r2']:+.3f}  "
                          f"base={base:+.3f}  opt={row['optimism']:+.3f}")
    return rows


def print_table(rows):
    hdr = (f"{'option':9s}{'target':36s}{'est':9s}{'n':>4s}"
           f"{'leaky':>9s}{'nested':>9s}{'base':>8s}{'optim':>8s}")
    print("\n" + "=" * len(hdr)); print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['option']:9s}{r['target'][:36]:36s}{r['estimator']:9s}{r['n']:>4d}"
              f"{r['leaky_r2']:>+9.3f}{r['nested_r2']:>+9.3f}{r['baseline_r2']:>+8.3f}"
              f"{r['optimism']:>+8.3f}")


if __name__ == "__main__":
    # Default = fast smoke test: vi option, OLS + plain RF.
    print("Quick smoke (vi option, OLS + RF_plain, RepeatedKFold 5x2).")
    print("For the full tuned grid use run_tuned_validation.py")
    cv = make_cv(splits=5, repeats=2)
    rows = run_grid(["vi"], ["OLS", "RF_plain"], cv=cv)
    print_table(rows)
