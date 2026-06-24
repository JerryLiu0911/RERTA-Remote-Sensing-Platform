# RERTA UAV Paper — Statistics Remediation Spec

Technical spec for what must change in the analysis and why. **No code is changed by this document.**
Each item: the problem, why it invalidates inference, the fix, and the expected effect on results.
Items are ordered by severity. References point to functions/files in the repo as of review.

---

## 1. Feature-selection leakage (CRITICAL — fix first)

**Where:** `main.py` calls `statistical_modelling.smart_feature_selection_pipeline(merged_df, target,
all_features)` on the **entire** dataset; the selected features are then passed to
`random_forest_regression(...)`, which performs its **own** train/test split internally
(`train_test_split`, `test_size=0.2`). The same leakage applies to OLS/LMM paths.

**Why it's wrong:** Feature ranking uses Spearman correlations with the target computed on all rows,
including the rows later used as the test set (`smart_feature_selection_pipeline` Stage 1 and Stage 3
both rank by `merged_df[target].corr(...)`). The held-out R² is therefore optimistically biased; the
"test" set is not truly held out. With n ≈ 40 and many candidate features, the inflation can be large.

**Fix:** Wrap selection + fitting in a single cross-validation loop so that, for each fold, feature
selection sees only that fold's training rows. Use nested CV (outer loop = performance estimate, inner
loop = selection + RF hyperparameter tuning). Given small n, prefer repeated k-fold or
leave-one-out for the outer estimate. Report the cross-validated R² distribution, not a single split.

**Expected effect:** Reported R² will **drop**, possibly to near zero for several targets. This is the
correct, honest number and is the foundation of the paper's argument.

---

## 2. Unit of analysis / pseudoreplication (CRITICAL)

**Where:** `gis.zonal_statistics` pools every pixel into `region_data[region_name].extend(clipped_data)`;
treatment boxplots (`create_boxplot_from_data`, `create_distribution_plots_from_data`) and any test using
`mannwhitneyu` (imported in `gis.py`) then operate on thousands of pixels per treatment.

**Why it's wrong:** Pixels within a plot are spatially autocorrelated, not independent replicates. The
experimental unit is the **vegetation plot** (n ≈ 12 per treatment), not the pixel (n ≈ thousands).
Pixel-level tests produce vanishingly small p-values that reflect pixel count, not ecological effect —
textbook pseudoreplication.

**Fix:** For every treatment comparison and figure, reduce each plot to a single summary value first
(the per-buffer `mean`/`range`/`cv` already computed in `zonal_statistics` are correct), then compare
**across plots**. If a treatment-difference test is reported, use plot-level values with a
mixed/nonparametric test appropriate to ~12 units/group, and report effect sizes with CIs. Retire any
pixel-level significance test.

**Expected effect:** Apparent treatment differences will look weaker and honest; some "clear" separations
in Figs 4–6 may become non-significant. Acceptable — and consistent with the framing that discrimination
is "visible but modest."

---

## 3. Multiple comparisons / best-of reporting (HIGH)

**Where:** `main.py` loops over all targets × 3 feature options (`vi`, `bands`, `bands+vi`) × up to 3
models. Narrative highlights the best (canopy openness, RF, R² = 0.36).

**Why it's wrong:** Dozens of fits with no correction; the maximum R² is expected to be inflated by
selection across the grid. Reporting only the best cell is a garden-of-forking-paths problem.

**Fix:** (a) Pre-specify a **single primary target** and primary model before looking at results
(canopy openness is the natural choice given the DEM proxy, but name it in advance). (b) Report the
**full grid** as a supplementary table with CIs. (c) Apply a multiplicity-aware lens (e.g., treat the
grid as exploratory and say so; or correct primary tests). Frame secondary results as exploratory.

**Expected effect:** No single cherry-picked headline number; the *pattern* (uniformly weak) carries the
result, which is more robust and more honest.

---

## 4. Linear mixed model under-identification (HIGH)

**Where:** `statistical_modelling.linear_mixed_model` uses `groups=df['treatment']` — 4 levels (A–D).

**Why it's wrong:** Random-effect variance is poorly estimated with < ~5–6 groups; the reported
pseudo-R² = 0.318 and "15–25% variance reduction" are fragile and may not be reproducible. Also,
treatment is arguably a *fixed* design factor, not a random sample of treatments.

**Fix:** Re-specify the random effect at a level with enough units — plot or transect **nested within**
treatment (the design has many transects/plots). If the data structure cannot support a stable random
effect, **retire the LMM** and report OLS + RF only; do not present an under-identified LMM as a third
"complementary" model. State the choice explicitly.

**Expected effect:** Either a defensible variance-partitioned model, or a cleaner two-model story. Both
beat the current fragile LMM.

---

## 5. Power / detectability analysis (HIGH — additive, not a fix)

**Why needed:** Without it, the negative result is uninterpretable ("no effect" vs. "no power"). With it,
the negative result becomes the paper's contribution.

**What to compute:**
- Minimum detectable correlation at the study's n (≈ 40 plots overall; ≈ 12 per treatment) at α = 0.05,
  80% power. As a planning anchor: n = 40 → detectable r ≈ 0.43 (R² ≈ 0.18); n = 12 → detectable
  r ≈ 0.66. (Compute exactly for the final n per analysis; these are order-of-magnitude anchors.)
- Implication statement: relationships weaker than this ceiling are invisible by construction, so the
  observed R² < 0.4 is consistent with either true weak association or undetectable moderate association.
- For RF, note that flexible models need far more data than this for stable held-out estimates; present
  RF results as exploratory.

**Expected effect:** Converts "it didn't work" into "we could only have detected strong effects, and even
those were absent for most targets" — a quantitatively bounded, publishable negative result.

---

## 6. DEM / terrain confound (MEDIUM — caveat, partial fix)

**Where:** DTM was discarded (report §2.2); `gis.canopy_openness_proxy` thresholds the DEM at an
**absolute** value (`data < 35`); treatment elevation differs (Fig 5) but treatments occupy different
terrain along the river.

**Why it's wrong:** Absolute DEM/elevation conflates ground topography with canopy height, so
between-treatment DEM differences and the absolute-threshold canopy-openness proxy are confounded by
terrain. Any "structural complexity" interpretation of mean DEM is unsafe.

**Fix / caveat:** Restrict DEM-derived predictors to **within-buffer relative** metrics (`range`, `cv`),
which are less terrain-sensitive, and state plainly that no normalised CHM exists. Do not interpret
absolute DEM or the absolute-threshold proxy as canopy structure without terrain control. If feasible,
sensitivity-check by regressing DEM metrics on plot terrain; otherwise disclose as a limitation.

**Expected effect:** The one variable most likely to survive (canopy openness, predicted partly by a
DEM proxy) is presented with appropriate caution, reducing the risk of an over-claimed positive.

---

## 7. Temporal & spatial misalignment (ACCEPT — disclose, do not "fix")

**Status:** Outside the authors' control (data collected by others, varying protocols/dates). Not a
remediation item — a **bounding limitation** to be quantified and disclosed.

**What to do:** Build a per-variable table of UAV date (2025) vs. field collection window, with the gap
in years. Note the consolidation method used (`load_field_data` aggregates across timepoints — e.g.,
`max` for canopy openness, `mean` for others) and acknowledge it mixes a temporal signal into a single
value with no time term in the model. Treat as a limitation on inference, not a flaw to hide.

---

## Validation checklist (after remediation)
- [ ] No correlation/feature ranking touches held-out rows at any point.
- [ ] Every figure/test uses one value per plot, never pooled pixels.
- [ ] Full model grid reported with CIs; primary target named a priori.
- [ ] LMM either re-specified (nested) or removed with justification.
- [ ] Power/detectability figure or table present and referenced in Discussion.
- [ ] DEM predictors limited to relative within-buffer metrics; terrain caveat stated.
- [ ] Temporal-gap table in supplement; consolidation method disclosed.
- [ ] R² values reported are cross-validated, not single-split; expect them lower than current.
