"""
plot_level_discrimination.py
============================
D2 — the linchpin test. Does UAV separate the four restoration treatments at the
PLOT level (one value per vegetation plot, ~12/treatment), rather than the pooled
PIXEL level used in the original report (Figs 4-6, which pseudoreplicates to
thousands of pixels and overstates significance)?

For every per-plot UAV metric (structural = DEM range/cv/mean/canopy_openness;
textural = GLCM contrast/homogeneity/energy/correlation; spectral = VI mean/range/cv)
it runs a Kruskal-Wallis omnibus test across treatments A/B/C/D and reports:
  H, raw p, Benjamini-Hochberg FDR-adjusted p, eta^2 effect size, per-group n/median.

Interpretation hook: if STRUCTURAL/TEXTURAL metrics still separate treatments at
the plot level, the "static, coarse signal is recoverable" half of the asymmetry
argument holds. If everything washes out at n~12/group, the central framing must
change (see MANUSCRIPT_OUTLINE.md).

Caveats: Kruskal-Wallis treats plots as independent; the design is actually nested
(plots within transect/side/distance), so this is the standard plot-level screen,
not the final model — a mixed model with nesting is the rigorous follow-up. Pools
distance classes (BC/OPE/OPC) within treatment, as the report's figures did.

SAFETY: reads Data/*.gpkg READ-ONLY; writes only plot_level_discrimination_results.{csv,md}.
"""

import os
import sys
import types
import csv

try:
    import shap  # noqa: F401  (project import; unused on this path)
except Exception:  # noqa: BLE001
    sys.modules["shap"] = types.ModuleType("shap")

import numpy as np
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(__file__))
import nested_cv_feature_selection as engine

OUT_BASE = os.path.join(os.path.dirname(__file__), "plot_level_discrimination_results")
TEXTURE_KEYS = ("contrast", "homogeneity", "energy", "correlation")


def feature_type(name):
    if any(t in name for t in TEXTURE_KEYS):
        return "textural"
    if "DEM" in name:
        return "structural"
    return "spectral"


def eta_squared_H(H, n, k):
    """Eta-squared effect size for Kruskal-Wallis; clipped at 0."""
    return max(0.0, (H - k + 1) / (n - k)) if n > k else float("nan")


def run(option="vi"):
    import contextlib, io
    sources, _ = engine.OPTIONS[option]
    with contextlib.redirect_stdout(io.StringIO()):
        df, _, features = engine.reconstruct_dataset(sources)

    if "treatment" not in df.columns:
        raise RuntimeError("no 'treatment' column in reconstructed data")
    treatments = sorted(t for t in df["treatment"].dropna().unique())
    k = len(treatments)
    print(f"[option={option}] {len(df)} plots, {len(features)} metrics, "
          f"treatments={treatments} (n/group: "
          f"{', '.join(f'{t}={int((df.treatment==t).sum())}' for t in treatments)})")

    rows = []
    for feat in features:
        groups, ns, medians = [], [], []
        for t in treatments:
            vals = df.loc[df["treatment"] == t, feat].dropna().to_numpy()
            groups.append(vals); ns.append(len(vals)); medians.append(np.median(vals) if len(vals) else np.nan)
        if sum(ns) < k + 4 or any(n < 3 for n in ns):
            continue
        if all(np.ptp(g) == 0 for g in groups):  # all constant
            continue
        try:
            H, p = kruskal(*groups)
        except Exception:  # noqa: BLE001
            continue
        rows.append({
            "feature": feat, "type": feature_type(feat), "H": H, "p_raw": p,
            "eta2": eta_squared_H(H, sum(ns), k), "n": sum(ns),
            "medians": medians,
        })

    if not rows:
        print("No testable metrics.")
        return rows, treatments

    # Benjamini-Hochberg FDR across all metrics tested
    pvals = [r["p_raw"] for r in rows]
    _, p_adj, _, _ = multipletests(pvals, method="fdr_bh")
    for r, pa in zip(rows, p_adj):
        r["p_fdr"] = pa
    rows.sort(key=lambda r: r["eta2"], reverse=True)
    return rows, treatments


def report(rows, treatments, option):
    hdr = f"{'metric':26s}{'type':11s}{'H':>7s}{'p_raw':>9s}{'p_fdr':>9s}{'eta2':>7s}  medians(A..)"
    print("\n" + "=" * len(hdr)); print(hdr); print("-" * len(hdr))
    for r in rows:
        med = " ".join(f"{m:.2f}" if np.isfinite(m) else "na" for m in r["medians"])
        print(f"{r['feature'][:26]:26s}{r['type']:11s}{r['H']:>7.2f}{r['p_raw']:>9.3f}"
              f"{r['p_fdr']:>9.3f}{r['eta2']:>7.2f}  {med}")

    sig = [r for r in rows if r["p_fdr"] < 0.05]
    sig_struct_tex = [r for r in sig if r["type"] in ("structural", "textural")]
    print("\n--- summary ---")
    print(f"metrics tested            : {len(rows)}")
    print(f"significant (FDR < 0.05)  : {len(sig)}")
    print(f"  of which struct/textural: {len(sig_struct_tex)}  "
          f"-> {'ASYMMETRY HOLDS' if sig_struct_tex else 'WASHED OUT — reframe needed'}")
    if sig:
        print("  top separating metrics  : "
              + "; ".join(f"{r['feature']}(eta2={r['eta2']:.2f},FDR={r['p_fdr']:.3f})" for r in sig[:5]))

    # write outputs (new files only)
    with open(OUT_BASE + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["feature", "type", "H", "p_raw", "p_fdr", "eta2", "n"]
                   + [f"median_{t}" for t in treatments])
        for r in rows:
            w.writerow([r["feature"], r["type"], f"{r['H']:.3f}", f"{r['p_raw']:.4f}",
                        f"{r['p_fdr']:.4f}", f"{r['eta2']:.4f}", r["n"]]
                       + [f"{m:.4f}" if np.isfinite(m) else "" for m in r["medians"]])
    with open(OUT_BASE + ".md", "w") as fh:
        fh.write(f"# Plot-level treatment discrimination (D2) — option={option}\n\n")
        fh.write("Kruskal-Wallis across treatments at the **plot** level (one value per plot). "
                 "`eta2` = effect size; `p_fdr` = Benjamini-Hochberg adjusted.\n\n")
        fh.write(f"Treatments: {treatments}.  Significant (FDR<0.05): {len(sig)} / {len(rows)} metrics; "
                 f"structural/textural among them: {len(sig_struct_tex)}.\n\n")
        fh.write("| metric | type | H | p_raw | p_fdr | eta2 |\n|---|---|--:|--:|--:|--:|\n")
        for r in rows:
            fh.write(f"| {r['feature']} | {r['type']} | {r['H']:.2f} | {r['p_raw']:.3f} "
                     f"| {r['p_fdr']:.3f} | {r['eta2']:.2f} |\n")
    print(f"\nSaved: {OUT_BASE}.csv / .md")


if __name__ == "__main__":
    opt = sys.argv[1] if len(sys.argv) > 1 else "vi"
    rows, treatments = run(opt)
    if rows:
        report(rows, treatments, opt)
