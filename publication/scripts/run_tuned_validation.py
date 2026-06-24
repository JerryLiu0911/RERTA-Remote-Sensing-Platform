"""
run_tuned_validation.py
========================
Heavy driver that runs the FULL leakage-free grid using the tuned Random Forest
(the project's tune_random_forest) plus the OLS baseline, across all three
analysis options (vi / bands / bands+vi).

Outputs (NEW files only — existing Data/ is never touched):
  publication/scripts/nested_cv_results.csv
  publication/scripts/nested_cv_results.md

Run:
  python publication/scripts/run_tuned_validation.py                 # full: 5x3 CV, n_iter=20
  python publication/scripts/run_tuned_validation.py --repeats 2 --n-iter 8   # faster
  python publication/scripts/run_tuned_validation.py --options vi --estimators OLS RF_tuned

This is compute-heavy (tuned RF nested in repeated CV). Consider running it in
the background. Requires the project dependencies (incl. shap, a project import);
if shap is absent it is harmlessly stubbed, since this code path never calls it.
"""

import os
import sys
import types
import argparse
import csv

# shap is imported at the top of statistical_modelling but is unused on this code
# path (we never call random_forest_regression). Stub it only if not installed.
try:
    import shap  # noqa: F401
except Exception:  # noqa: BLE001
    sys.modules["shap"] = types.ModuleType("shap")

sys.path.insert(0, os.path.dirname(__file__))
import nested_cv_feature_selection as engine


def parse_args():
    p = argparse.ArgumentParser(description="Tuned, leakage-free CV grid for RERTA models.")
    p.add_argument("--options", nargs="+", default=["vi", "bands", "bands+vi"],
                   choices=list(engine.OPTIONS.keys()))
    p.add_argument("--estimators", nargs="+", default=["OLS", "RF_tuned"],
                   choices=list(engine.FACTORIES.keys()))
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--n-iter", type=int, default=20, help="RandomizedSearchCV iterations for tuned RF")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "nested_cv_results"))
    return p.parse_args()


def write_outputs(rows, out_base, meta):
    csv_path, md_path = out_base + ".csv", out_base + ".md"
    fields = ["option", "target", "estimator", "n", "leaky_r2", "nested_r2",
              "baseline_r2", "optimism", "mean_dim", "top_feats"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(md_path, "w") as fh:
        fh.write("# Leakage-free CV validation — tuned grid\n\n")
        fh.write(f"_{meta}_\n\n")
        fh.write("`nested_r2` is the honest out-of-sample estimate (transform re-fit per "
                 "fold). `leaky_r2` fixes the transform on all rows (current main.py). "
                 "`optimism = leaky - nested`. `baseline_r2` is a mean predictor.\n\n")
        fh.write("| option | target | est | n | leaky R² | nested R² | base R² | optimism |\n")
        fh.write("|---|---|---|--:|--:|--:|--:|--:|\n")
        for r in rows:
            fh.write(f"| {r['option']} | {r['target']} | {r['estimator']} | {r['n']} "
                     f"| {r['leaky_r2']:+.3f} | {r['nested_r2']:+.3f} "
                     f"| {r['baseline_r2']:+.3f} | {r['optimism']:+.3f} |\n")
    return csv_path, md_path


def main():
    args = parse_args()
    engine.RF_TUNE_N_ITER = args.n_iter
    cv = engine.make_cv(splits=args.splits, repeats=args.repeats)
    meta = (f"options={args.options}, estimators={args.estimators}, "
            f"CV=RepeatedKFold({args.splits}x{args.repeats}), RF n_iter={args.n_iter}")
    print("=" * 72)
    print("TUNED LEAKAGE-FREE VALIDATION GRID")
    print(meta)
    print("Reads Data/*.gpkg read-only. Writes only nested_cv_results.{csv,md}.")
    print("=" * 72)

    rows = run = engine.run_grid(args.options, args.estimators, cv=cv, verbose=True)
    engine.print_table(rows)
    csv_path, md_path = write_outputs(rows, args.out, meta)
    print(f"\nSaved:\n  {csv_path}\n  {md_path}")


if __name__ == "__main__":
    main()
