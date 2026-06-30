# RERTA UAV Paper — Section-by-Section Outline

**Working title (candidates):**
- "Visible but not predictive: UAV imagery captures restoration structure yet fails to predict
  ecological function in oil palm riparian buffers"
- "A reality check for UAV monitoring of tropical riparian restoration: structural signal without
  functional prediction"

**Central argument (the through-line every section serves):**
UAV metrics easily separate restoration treatments because the treatments are *visibly* different by
design (1). But the same UAV metrics do not predict ecological *function* — canopy openness, soil
erosion, seed predation, frog biodiversity (2). We show this gap is **informative**, not merely an
artefact of small n, by quantifying the effect sizes the study could have detected (3), and we disclose
the data-alignment constraints that bound the result (4). Deliverable: a reproducible workflow and a
calibrated expectation for the field (5).

**Central thesis (post-validation, narrow & defensible — the asymmetry argument):**
UAV imagery robustly recovers the **static, coarse-scale** signature of the restoration manipulation —
at the *plot* level, vegetation texture and spectral-range metrics separate the four treatments with
large effects (η² up to 0.65; 20/39 metrics FDR<0.05). Yet the **same plots, same data, same n** yield
no out-of-sample predictive skill for **dynamic, fine-scale** ecological *function* (canopy openness
marginal; erosion, seed removal at/below baseline). These differ not because the data are arbitrarily
"good for one, bad for the other" but because static/coarse signals are **robust** to the temporal and
spatial misalignment that characterises operational UAV–field integration, while dynamic/fine-scale
function is **exactly what that misalignment attenuates** (supporting evidence: predictability tracks the
*temporal stability* of the measurement, not ecological function as such).

**Scope (do not overreach):** we frame the negative as **operational** — UAV did not predict function
under realistic, non-contemporaneous data conditions — **not** as a structure–function *decoupling in
nature*, which these data cannot establish. The decoupling hypothesis is explicitly **retired** as the
least supportable explanation. The contribution is a rigorous, reproducible characterisation of this
asymmetry plus a data-collection prescription (contemporaneous, co-located capture) for a fair test.

---

## What the leakage-free validation established (June 2026)

Run via `publication/scripts/run_tuned_validation.py` (nested CV, feature selection re-fit inside each
fold). **Numbers below are from a reduced-CV exploratory pass (RepeatedKFold 5×2, RF n_iter=8) and are
AI-run; treat as provisional pending the author's full-settings re-run and verification (see Next
Steps).** Saved in `publication/scripts/nested_cv_results.{csv,md}`.

1. **The report's RF R² = 0.36 (canopy openness) does not reproduce.** Honest out-of-sample R² is
   ≈ +0.04 (tuned RF) to +0.18 (OLS). The 0.36 was a single small-test-split artefact.
2. **Canopy openness is the lone qualified positive** — honest nested R² ≈ **0.14–0.18 (OLS, vi and
   bands+vi)**, above the mean baseline (≈ −0.08) but right at the detectability ceiling (n=48 ⇒ ~0.16).
   Selected on `canopy_openness_DEM` in every fold — but this is a **terrain/treatment confound, not
   proxy validation**: the proxy tracks terrain (r≈−0.76) more than field openness (+0.50); controlling
   for terrain/treatment collapses the link to 0.22/0.30 (see §4.1b/4.1d). Even the lone positive
   dissolves under scrutiny.
3. **Every functional variable is unpredictable** — erosion (Circle/Harvesting/Windrow/average, n=32)
   and seed removal (n=48): nested R² ≤ 0, at or below baseline, across all three feature sets and both
   models. Isolated small RF positives (e.g. `average_change` bands+vi +0.14) flip sign across feature
   sets and carry large leakage optimism ⇒ non-reproducible noise.
4. **OLS ≥ tuned RF wherever it matters.** Flexible models hurt at n = 32–48 — a quotable
   "do-not-over-model" result.
5. **Leakage/instability is worst for RF + selection** (optimism −0.20 to +0.24, sign-flipping) and
   small/stable for OLS — itself a methodological finding worth a figure.
6. **Treatment discrimination survives at the plot level (D2 —
   `plot_level_discrimination.py`).** Kruskal–Wallis across treatments on one value per plot,
   FDR-corrected: **20/39 metrics significant, 11 structural/textural, η² up to 0.65** (`range_GLI`),
   Clre/ReNDVI texture η²≈0.44–0.46. *Driven by vegetation texture + spectral range, NOT terrain*:
   `mean_DEM` separates (η²=0.37) but is terrain-confounded, while `range_DEM`/`cv_DEM` do **not**
   separate — so the structural claim rests on texture/spectral-range, sidestepping the DSM confound.
   Caveat: KW treats plots as independent (design is nested); a mixed model is the rigorous follow-up,
   but η² of this size is unlikely to vanish. Note some specific report claims do not survive (e.g.
   `homogeneity_GLI`, FDR=0.40) — revise those.

---

## Abstract
Keep the existing honest structure. Sharpen to lead with the gap: imagery captures the manipulation's
structure (expected), but under leakage-free cross-validation no functional target exceeds a mean
baseline; the one target with any signal (canopy openness, R² ≈ 0.15 via a DEM structural proxy) sits at
the detectability ceiling. Power analysis shows only strong effects (r ≳ 0.40 at n = 48) were
detectable — and even that ceiling applies to data already attenuated by temporal/spatial misalignment,
so we frame the negative as *operational* (UAV did not predict function under these realistic,
non-contemporaneous conditions), not as evidence the underlying relationship is absent. Note that
simpler models (OLS) matched or beat tuned random forests. End on the workflow + methodological
guidance.

## 1. Introduction
- Oil palm expansion → riparian buffers as RSPO-aligned restoration (existing text is strong; keep).
- UAV remote sensing promise for restoration monitoring; the *implicit assumption* the field makes:
  spectral/structural signal ⇒ ecological function can be inferred. **Name this assumption — it is what
  you test.**
- Gap: empirical tests of that assumption in operational tropical oil palm are scarce.
- **Aim (unchanged from the project):** predict field-measured ecological function from UAV indices,
  structure, and texture across four restoration treatments. State up front this is a *predictive* test,
  with treatment discrimination as a validation step.

## 2. Methods
*(Mostly exists; complete and tighten.)*
- 2.1 Site & design (treatments A–D, plots, transects, distances, timepoints). Keep Fig 1–2.
- 2.2 UAV acquisition & processing. **Complete the Appendix** here or in supplement: Metashape SfM
  parameters, RTK, calibration, DEM/DTM handling. **State explicitly that DTM was discarded and DEM is a
  surface model** — and what that forecloses (no clean CHM).
- 2.3 Ecological datasets (Table 1; keep). Add the per-variable **temporal-gap** column (UAV 2025 vs.
  field collection date) — turns a hidden weakness into transparent method.
- 2.4 Vegetation indices (Table 2; keep).
- 2.5 Research design — **revise to reflect remediated analysis:**
  - Unit of analysis = **vegetation plot** (state this prominently).
  - Feature selection performed **inside cross-validation folds** (no leakage).
  - Full model grid pre-specified; primary target named in advance.
- 2.6 Statistical modelling: OLS (baseline), RF (non-linear); LMM *only if* re-specified with
  plot/transect nesting, otherwise drop. State that performance is the **pooled out-of-fold R²** from
  repeated CV with selection/PCA re-fit per fold, benchmarked against a **mean baseline**, and that a
  **leaky vs. nested** comparison is reported to quantify selection optimism. Add **power analysis**
  method here.

## 3. Results
- 3.1 **Treatment discrimination (the bridge — now the load-bearing contrast).** Report the **plot-level**
  Kruskal–Wallis result (D2): 20/39 metrics separate treatments at FDR<0.05, large effects (η² up to
  0.65 for `range_GLI`; Clre/ReNDVI texture η²≈0.44–0.46). Revised Figs 4–6 at plot level (one value per
  plot), not pixel-pooled. Frame as **expected** (treatments differ visibly by design) — its job is to
  establish that the pipeline *can* carry a signal, making the prediction failure in 3.2 a property of
  the task, not the pipeline. **Lean on texture + spectral range, not DEM:** `mean_DEM` separates but is
  terrain-confounded, and `range_DEM`/`cv_DEM` do not separate — so vegetation texture/spectral range
  carries the structural claim and the DSM/terrain confound is sidestepped, not hand-waved. Drop the
  `homogeneity_GLI` claim (does not survive); use Clre/ReNDVI texture instead.
- 3.2 **Predicting ecological function (the actual result).** Full grid table (target × feature set ×
  model) of honest nested R² vs. baseline, with the leaky column alongside. Lead with the *pattern*:
  canopy openness is the only target clearing baseline (~0.15, OLS), every functional target sits at or
  below baseline. Explicitly note the report's 0.36 does not reproduce. Headline figure: honest R² per
  target with the detectability ceiling overlaid (see 3.3).
- 3.2b **Model complexity and selection leakage (methodological result).** Two points, each a small
  figure/panel: (i) **OLS ≥ tuned RF** across targets — added flexibility did not help at n = 32–48;
  (ii) **leaky − nested optimism** is small/stable for OLS but large and sign-inconsistent for
  RF + selection — a concrete cautionary illustration for small-n UAV-ecology modelling. Report
  feature-selection **stability** (e.g. `canopy_openness_DEM` chosen every fold) as a diagnostic.
- 3.3 **Power / detectability (the keystone).** Minimum detectable effect at 80% power, α = 0.05:
  **n = 48 ⇒ r ≈ 0.40, R² ≈ 0.16; n = 32 ⇒ r ≈ 0.48, R² ≈ 0.23; per-treatment n ≈ 12 ⇒ r ≈ 0.66.**
  (Correlation-based ⇒ an *optimistic* bound; the multivariate/RF models had less power still.) The lone
  positive (canopy openness, ~0.15) sits *at* the ceiling; functional targets fall well below it.
  **Scope of the conclusion (do not overreach):** the power ceiling applies to the *observed* effect in
  these data, which is itself *attenuated* by measurement limitations — temporal mismatch (2025 imagery
  vs 2017–2024 field data) and positional error (~2 m, no field GPS) both bias correlations toward zero
  before sampling noise enters. So we can say strong links *as expressed in this dataset* are absent, but
  we **cannot** separate a genuine absence from a real relationship masked by misalignment. The result is
  bounded by **two** limits — sampling power *and* data alignment — and only the first is what a power
  analysis addresses. Frame the negative as *operational* ("UAV did not predict function under these
  realistic, non-contemporaneous data conditions"), not as a claim about the underlying ecology.

## 4. Discussion *(currently missing — write in full)*
- 4.1 **The asymmetry, not a decoupling.** Lead here. The pipeline recovers the *static, coarse* signal
  (treatment identity; D2, η² up to 0.65) but not the *dynamic, fine-scale* functional variables — from
  the same plots and n. Interpret via **differential exposure to the data limitations**, not ecology:
  treatment identity is time-invariant and block-scale ⇒ immune to temporal mismatch and to ~2 m
  positional error; function is dynamic and plot-scale ⇒ attenuated by both. Candidate causes of the
  prediction failure, weighted: (a) temporal mismatch (2025 imagery vs 2017–2024 field, tree growth);
  (b) positional error vs buffer size; (c) resolution/penetration under dense canopy; (d) limited power;
  (e) **[retired/least-supported]** genuine structure–function decoupling — explicitly *cannot* be
  distinguished from (a)–(d) and should not be claimed. A possible "predictability tracks temporal
  stability" gradient (volatile targets — seed removal, frogs — most null) is *suggestive of attenuation*
  but **weak**: do not lean on canopy openness as the slow-changing exemplar, because its apparent signal
  is a terrain/treatment confound (4.1b), not a genuine slow-structure→function link.
- 4.1a **Why the contrast is informative, not contradictory.** Because discrimination succeeds, gross
  georeferencing/processing failure is excluded — the pipeline carries a signal; it just doesn't carry a
  *functional, contemporaneous* one. This is the crux that lets the negative result mean something
  specific rather than "bad data".
- 4.1b **The lone positive is a terrain artefact, not proxy validation.** Canopy openness — the only
  target nominally clearing baseline — is **not** structure-predicts-structure. The DEM-derived
  `canopy_openness` proxy tracks *terrain elevation* far more than the field measurement (Pearson
  r ≈ −0.76 with mean DEM vs +0.50 with field openness), and most of its apparent agreement with field
  openness is a **spatial confound**: controlling for terrain drops proxy↔field 0.50→0.22; controlling
  for treatment, 0.50→0.30; the within-treatment correlation is weak and inconsistent (mean ≈ 0.28;
  *negative* in one treatment). The path is `terrain ← (location of) treatment → canopy openness`. So
  even the single apparent positive dissolves into a terrain/treatment confound — which **strengthens**
  the negative (no genuine direct UAV→function signal) and motivates 4.1d.
- 4.1c **Complexity did not help.** OLS matched or beat tuned RF, and RF + selection showed large,
  unstable leakage optimism. At this n, flexible models add variance, not skill — a transferable caution
  for the UAV-ecology literature, which increasingly defaults to RF/deep models on small field samples.
- 4.1d **The DSM/DTM/CHM problem (transferable methodological lesson).** What canopy structure *requires*
  is a CHM (= DSM − DTM); we have only the **DSM**, because dense closed canopy starves the optical-SfM
  point cloud of ground returns, making the DTM unreliable — a fundamental limit of photogrammetry
  (vs LiDAR), **not** an analysis error. Without ground normalisation, an *absolute-threshold* DSM metric
  silently becomes a **terrain index** (the mechanism behind 4.1b). Generalise it: practitioners reaching
  for SfM "canopy height" under dense tropical canopy may be measuring topography. Mitigations and their
  limits — (i) *relative* within-buffer DSM metrics (range/cv) avoid the absolute-elevation confound but
  are weak here (`range_DEM`/`cv_DEM` do not separate treatments); (ii) DSM detrending approximates a
  pseudo-ground but fails under fully closed canopy; (iii) the proper fix is a **LiDAR-derived DTM → true
  CHM**. **Future work:** the site's LiDAR exists but is **unprocessed** — processing it for a reliable
  DTM/CHM is the clear next step to test whether genuine canopy *structure* (not terrain) carries any
  functional signal. This is the concrete instance that makes the DSM limitation *content*, not a caveat.
- 4.2 **Why discrimination ≠ prediction.** The conceptual core: separating treatments (a coarse,
  visible contrast) is far easier than regressing a continuous functional variable; reviewers and the
  field conflate these. State it.
- 4.3 **Implications for UAV restoration monitoring.** Calibrated expectations: what UAV *can* do here
  (detect/track the visible manipulation, structural change over time) vs. what it *cannot yet* do
  (stand in for functional field measurement). Design recommendations: contemporaneous UAV+field
  capture, georeferenced field plots, larger replication, multi-temporal UAV.
- 4.4 **Limitations** (honest, immovable): n; single site (Kandista excluded — say why); temporal and
  spatial misalignment outside the authors' control; **DSM/terrain confound (the canopy structure
  channel is compromised — see 4.1b/4.1d, the paper's most concrete limitation)**; known data errors.
  Frame as
  bounding conditions on inference, not excuses.

## 5. Conclusion *(currently missing — write)*
UAV captures the visible structure of restoration but does not, under these conditions, predict
ecological function; the negative result is bounded and quantified, not null. A reproducible pipeline
and concrete design guidance for future UAV–field studies are the lasting outputs.

## Appendix / Supplement *(currently just a link — build out)*
- SfM/Metashape full workflow + parameter table.
- Pipeline description (modules: load_field_data, coordinate_extraction, gis.zonal_statistics,
  statistical_modelling) with the repo as archived supplement (Zenodo DOI).
- Full model-grid table; per-variable temporal-gap table; data dictionary for field datasets.

---

## Claim → evidence map (keep every claim tethered)
| Claim | Evidence | Status / risk |
|---|---|---|
| Treatments are separable by UAV (static/coarse signal) | **Plot-level KW: 20/39 metrics FDR<0.05, η² up to 0.65** (D2) | **Supported** (provisional; nested mixed-model is the rigorous follow-up). Frame as "expected", lean on texture/spectral-range not DEM |
| Function is not predictable from UAV | Nested-CV grid: all functional targets ≤ baseline | **Supported** (provisional, reduced CV). Confirm with full-settings re-run |
| The report's RF R²=0.36 does not reproduce | Nested RF ≈ 0.04; single-split artefact | **Supported** — strengthens the negative |
| Lone positive = structure→structure, not ecology | Canopy openness ~0.15 (OLS) via `canopy_openness_DEM` | **Supported**; name the circularity. Risk: do not over-claim it as success |
| Model complexity does not help at small n | OLS ≥ tuned RF; RF+selection optimism unstable | **Supported** (provisional) — needs full-settings confirmation |
| The negative result is informative (operational, bounded) | Power/MDES + attenuation argument | Spine — but state scope: power bounds the *observed* (attenuated) effect, not the true relationship. Claim "no prediction under these data conditions", NOT "no relationship exists" |
| Asymmetry: static/coarse recoverable, dynamic/fine not (under these data) | D2 (discrimination) vs nested-CV grid (prediction), same plots/n | **Core claim** — supported by the contrast; frame as differential exposure to misalignment |
| ~~Structure–function decoupling in nature~~ | — | **RETIRED** — cannot be separated from temporal/spatial/power causes; do not claim |
| Workflow is reproducible | Archived repo + Appendix | Low; pending raster data-availability decision |
