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

---

## Abstract
Keep the existing honest structure. Sharpen to lead with the gap: imagery captures the manipulation's
structure (expected), but R² < 0.4 for all functional targets; power analysis shows only strong effects
(r ≳ 0.43) were detectable, so the negative result is a genuine ceiling under these conditions, not just
noise. End on the workflow + methodological guidance.

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
- 2.6 Statistical modelling: OLS (baseline), RF (non-linear), and LMM *only if* re-specified with
  plot/transect nesting; otherwise drop. Add **power analysis** method here.

## 3. Results
- 3.1 **Treatment discrimination (the bridge, framed as expected).** Plot-level boxplots (revised Figs
  4–6). One or two sentences: "As anticipated, treatments separate on structure/texture because they
  differ visibly by design; this validates that UAV captures the manipulation." Do **not** oversell.
  Flag the DEM/terrain caveat where structure is interpreted.
- 3.2 **Predicting ecological function (the actual result).** Full grid table with honest R²/RMSE/CI
  for every target × feature set × model. Lead with the *pattern* (uniformly weak), not the best cell.
- 3.3 **Power / detectability (the keystone).** Effect-size sensitivity: at n ≈ 40 (and per-treatment
  n ≈ 12), the minimum correlation detectable at 80% power ≈ 0.43 (R² ≈ 0.18). Therefore weak–moderate
  true relationships are *invisible by construction*. This is what makes the negative result mean
  something.

## 4. Discussion *(currently missing — write in full)*
- 4.1 **Structure visible, function not predictable.** Interpret the gap. Candidate explanations,
  weighted: (a) genuine structure–function decoupling at this scale; (b) temporal mismatch (tree growth
  between 2025 imagery and field data up to ~3 yr); (c) spatial/positional uncertainty vs. buffer size;
  (d) resolution/penetration limits under dense canopy (river/understory); (e) insufficient power.
  Be explicit about which you can and cannot rule out.
- 4.2 **Why discrimination ≠ prediction.** The conceptual core: separating treatments (a coarse,
  visible contrast) is far easier than regressing a continuous functional variable; reviewers and the
  field conflate these. State it.
- 4.3 **Implications for UAV restoration monitoring.** Calibrated expectations: what UAV *can* do here
  (detect/track the visible manipulation, structural change over time) vs. what it *cannot yet* do
  (stand in for functional field measurement). Design recommendations: contemporaneous UAV+field
  capture, georeferenced field plots, larger replication, multi-temporal UAV.
- 4.4 **Limitations** (honest, immovable): n; single site (Kandista excluded — say why); temporal and
  spatial misalignment outside the authors' control; DSM/terrain confound; known data errors. Frame as
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
| Claim | Evidence | Risk if challenged |
|---|---|---|
| Treatments are separable by UAV | Plot-level Figs 4–6 | Low — but must say "expected", not novel |
| Function is not predictable from UAV | Full grid, all R² < 0.4 w/ CV | Medium — depends on leakage fix holding |
| The negative result is informative | Power analysis, detectable r ≈ 0.43 | Low if power calc is sound — this is the spine |
| Structure–function decoupling plausible | Discussion synthesis | High — hedge; cannot fully separate from power/temporal causes |
| Workflow is reproducible | Archived repo + Appendix | Low |
