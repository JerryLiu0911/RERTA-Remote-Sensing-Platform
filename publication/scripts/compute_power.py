"""
compute_power.py
================
Minimum detectable effect size (MDES) for a Pearson correlation, via the Fisher-z
approximation, at the sample sizes used in the study. This is a *design
calculation*, not a hypothesis test — it states the smallest correlation the study
had a given chance of detecting. Writes power_statement.md for the write-up.

  detectable z_r = (z_{1-a/2} + z_power) / sqrt(n - 3);  r = tanh(z_r);  R^2 = r^2

Correlation-based ⇒ an OPTIMISTIC bound: multivariate/RF models have less power
still, and any true effect is further attenuated by temporal/spatial misalignment
(so this bounds the *observed* effect, not the underlying relationship).
"""
import os
from math import tanh, sqrt
from scipy.stats import norm

ALPHA, POWER = 0.05, 0.80
SAMPLES = {
    48: "canopy openness, seed removal (all plots)",
    32: "erosion targets (Buffer-Core + Oil-Palm-Core plots)",
    12: "within a single treatment",
}


def mdes(n, alpha=ALPHA, power=POWER):
    z = (norm.ppf(1 - alpha / 2) + norm.ppf(power)) / sqrt(n - 3)
    r = tanh(z)
    return r, r * r


def main():
    rows = [(n, *mdes(n), lbl) for n, lbl in SAMPLES.items()]
    print(f"MDES at {int(POWER*100)}% power, alpha={ALPHA} (two-sided):")
    for n, r, r2, lbl in rows:
        print(f"  n={n:3d}  r>={r:.2f}  R^2>={r2:.2f}   ({lbl})")

    out = os.path.join(os.path.dirname(__file__), "power_statement.md")
    with open(out, "w") as fh:
        fh.write("# Power / minimum detectable effect size\n\n")
        fh.write(f"At {int(POWER*100)}% power and alpha = {ALPHA} (two-sided), Fisher-z approximation.\n\n")
        fh.write("| n | context | detectable r | detectable R^2 |\n|--:|---|--:|--:|\n")
        for n, r, r2, lbl in rows:
            fh.write(f"| {n} | {lbl} | {r:.2f} | {r2:.2f} |\n")
        n48 = dict((n, (r, r2)) for n, r, r2, _ in rows)[48]
        n32 = dict((n, (r, r2)) for n, r, r2, _ in rows)[32]
        fh.write(
            "\n**Ready-to-adapt statement:**\n\n"
            f"> With n = 48 plots, the study had 80% power (alpha = 0.05) to detect a "
            f"correlation of r ≈ {n48[0]:.2f} (R² ≈ {n48[1]:.2f}); for the erosion targets "
            f"(n = 32) the threshold rises to r ≈ {n32[0]:.2f} (R² ≈ {n32[1]:.2f}). Relationships "
            f"weaker than this were undetectable by construction. Because this correlation-based "
            f"bound is optimistic (multivariate and random-forest models have lower power, and any "
            f"true effect is further attenuated by the temporal and spatial misalignment of the "
            f"data), the ceiling on the *observed* effect is at least this high — so the study can "
            f"exclude only strong associations as expressed in these data, not weak associations "
            f"nor the underlying relationship.\n"
        )
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
