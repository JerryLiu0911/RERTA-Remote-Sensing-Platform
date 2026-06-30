# RERTA UAV Paper — Action Plan

**Target framing (current — operational asymmetry):** UAV recovers the **static, coarse** manipulation
signal (plot-level treatment discrimination confirmed: 20/39 metrics FDR<0.05, η² up to 0.65, via
vegetation texture/spectral range) but does **not** predict **dynamic, fine-scale** ecological function
(honest out-of-sample R² ≤ baseline; canopy openness marginal ~0.15). Framed as differential exposure to
temporal/spatial misalignment — **operational**, not a structure–function decoupling in nature (that
claim is retired). Contribution = rigorous characterisation of the asymmetry + a fair-test prescription
(contemporaneous, co-located capture).

**Target journal:** the methodological/operational pole makes **Frontiers in Forests & Global Change**
(sister RERTA papers — Woodham 2019, Drewer 2024), **Drones (MDPI)**, or **PLOS ONE** as strong as
**Scientific Reports**. Decide with supervisor.

**One-line thesis to defend:**
> "From the same plots, high-resolution UAV imagery robustly recovers which restoration treatment a plot
> belongs to, yet does not predict ecological function under realistic, non-contemporaneous field-data
> conditions — an asymmetry driven by exposure to temporal/spatial misalignment, not by a structure–
> function decoupling. We quantify it and prescribe the data regime needed to test the relationship fairly."

---

## CURRENT STATUS (June 2026): statistics frozen → writing phase

Per agreed scope, **no new statistical tests** beyond what is done. Remaining stats work is *finalisation
only*:
- [ ] **Finalise prediction numbers** — full-settings re-run (`run_tuned_validation.py --repeats 5
  --n-iter 20`); verify and own the output (Lane 1). *(Running now; refreshes `nested_cv_results.md`.)*
- [x] **Power statement** — `compute_power.py` → `power_statement.md` (n=48: r≥0.39/R²≥0.16; n=32:
  r≥0.48/R²≥0.23).
- [x] **Plot-level discrimination (D2)** — `plot_level_discrimination.py` → results saved.
- [ ] **Temporal-gap table** — descriptive (imagery vs field dates); not a test.

**Dropped to honour minimum-stats:** nested mixed-model for D2 (keep KW + a disclosed-nesting sentence);
temporal-stability gradient (make it a qualitative Discussion argument, not a test).

Consolidated numbers for the write-up: **`publication/WRITEUP_NUMBERS.md`**.

### Writing checklist (you write prose; AI copy-edits only — Lane 1)
- [ ] **Abstract** — re-cast to operational asymmetry; drop "R²<0.4 / ecology" claims.
- [ ] **Methods** — add: unit of analysis = plot; feature selection inside CV folds; DEM-is-a-DSM (DTM
  discarded); SfM/Metashape params; power method. Complete the bare-link Appendix.
- [ ] **Results — revise existing** — drop the 0.36; report honest nested R² + baseline; replace
  pixel-pooled Figs 4–6 with plot-level versions; add the D2 discrimination result; add power/MDES.
- [ ] **Results — model-complexity note** — OLS ≥ tuned RF; leaky-vs-nested optimism (small fig).
- [ ] **Discussion (new)** — the asymmetry (static/coarse recoverable vs dynamic/fine not); differential
  exposure to misalignment; retire decoupling; limitations (n, single site, misalignment, DSM); design
  recommendations.
- [ ] **Conclusion (new)** — operational bounded negative + reproducible workflow + fair-test prescription.
- [ ] **Data/code availability** — repo + Zenodo DOI; raster-deposit decision (see below).

---

## Fix vs. Accept (the dividing line)

| Issue | In your control? | Decision |
|---|---|---|
| Feature selection outside CV (leakage) | Yes | **FIX** — non-negotiable validity bug |
| Pixel pooling for treatment stats (pseudoreplication) | Yes | **FIX** — analyse at plot level |
| Best-of-grid reporting, no correction | Yes | **FIX** — report full grid + correction |
| LMM random effect = 4 treatments only | Yes | **FIX** — nest plot/transect, or drop LMM claim |
| No power analysis | Yes | **FIX** — add; it is what makes the negative result mean something |
| Small n (≈32–48 plots) | No | **ACCEPT** + quantify via power analysis |
| Temporal mismatch (UAV 2025 vs field 2017–2024) | No | **ACCEPT** — honest limitation, quantified per variable |
| Spatial offset (~2 m), no true GPS on many datasets | No | **ACCEPT** — honest limitation |
| Kandista has baseline-only field data | No | **ACCEPT** — state why excluded |
| DEM is a DSM (DTM discarded) → terrain confound | Partly | **CAVEAT** structural claims; do not interpret absolute DEM as canopy |

---

## Phase 1 — Statistical remediation (blocking; see STATS_REMEDIATION.md)
Priority order:
1. **Move all feature selection inside the CV loop.** Re-run every predictive model. Expect reported
   R² to drop — that is correct and expected. (`smart_feature_selection_pipeline` currently runs on the
   full `merged_df` before the split in `random_forest_regression`.)
2. **Fix unit of analysis.** Every treatment comparison and figure uses **plot-level** values
   (one row per vegetation plot), never pooled pixels. Remove/replace any pixel-level significance test
   (the `mannwhitneyu` usage in `gis.py`).
3. **Report the full model grid** (targets × {vi, bands, bands+vi} × {OLS, LMM, RF}) in a supplementary
   table with confidence intervals; stop foregrounding the single best R².
4. **Power analysis** (detectable effect size at n≈40, and per-treatment n≈12). This is the spine of the
   "informative negative result" argument.
5. **LMM:** re-specify random effects as plot/transect nested in treatment, or retire the LMM as
   under-identified (4 groups) and keep OLS + RF.

## Phase 2 — Re-analysis & figures
- Regenerate all predictive results post-remediation; record honest R²/CI per model.
- Keep Figs 4–6 (treatment boxplots) but **plot-level**, reframed as "the manipulation is visible."
- New figure: **observed vs. detectable effect size** (power curve) — the paper's keystone.
- Optional: if any *structural* interpretation is retained, add a terrain caveat or sensitivity check;
  otherwise restrict DEM use to within-buffer `range`/`cv` and state the DSM/terrain confound plainly.

## Phase 3 — Manuscript completion (see MANUSCRIPT_OUTLINE.md)
- Write the **missing half**: Discussion, Limitations, Conclusion. Currently the report ends mid-Results.
- Replace the bare-link Appendix with: SfM/Metashape workflow, parameter table, and a methods-grade
  description of the pipeline (the repo is the supplement, not the methods).
- Tone pass: downgrade "impressive/substantial/synergistic" to match R² < 0.4 on n ≈ 40.

## Phase 4 — Submission readiness
- Data & code availability statement (repo + archived DOI via Zenodo release).
- Per-variable temporal-gap table (UAV date vs. field collection date) as a supplement.
- Author contributions, RSPO/permits, ethics (frog surveys), funding (UROP).

## AI / LLM use — Scientific Reports (Springer Nature) compliance

Scientific Reports follows Springer Nature editorial policy on AI. Rules that bind this manuscript:

1. **No AI authorship.** LLMs cannot be authors (cannot take accountability). Authorship = human
   contributors only.
2. **Disclose substantive AI use; copy-editing is exempt.**
   - *No declaration needed* if AI only improves readability/grammar/formatting of **human-written** text.
   - *Declaration required* if AI helped **generate text, data analysis, or content** ("generative
     editorial work / autonomous content creation") — declare in **Methods** and **Acknowledgements**.
3. **Human accountability for the final version is mandatory.** Authors must verify every fact, number,
   statistical claim, and **citation** (LLMs fabricate/misattribute references — check all ~30).
4. **AI-generated *images* are essentially prohibited** (narrow exceptions, must be labelled
   "AI-generated"). Note: data-derived plots (matplotlib/seaborn) are **not** AI-generated images and
   are fine. Do **not** use AI image generators for the schematic/map figures.
5. **Peer-review (for reviewing others):** don't upload manuscripts to AI tools; disclose any AI use in
   review reports.

**Lane decision: LANE 1 — copy-edit only (CHOSEN).** No AI disclosure required.
- The authors write **all** manuscript prose. AI is used only to polish grammar/readability/formatting
  of author-written text and to advise on structure, statistics, and planning.
- AI will **not** draft manuscript sections (no ghost-written Discussion, power-analysis prose, etc.).
- Planning docs in `publication/` are internal scaffolding and are **never pasted** into the manuscript.
- Still mandatory: human verification of every number, claim, and citation.

*(Lane 2 — AI-assisted drafting — was not chosen; if ever revisited, it requires a disclosure statement
in Methods + Acknowledgements.)*
- Cover letter positioning the negative result as informative, not null.

---

## Rough sequencing (effort, not calendar — adjust to your availability)
- **P1 stats fixes:** small code changes, large interpretive impact. Do first.
- **P2 re-analysis:** mechanical once P1 lands.
- **P3 writing:** the largest block; the Discussion is where acceptance is won or lost.
- **P4 polish:** short.

## Open decisions to revisit after P1 results land
- If post-leakage-fix prediction is *uniformly* near-zero R² → lean harder into the
  structure/function-decoupling story (cleaner paper).
- If one variable (likely canopy openness, given the DEM proxy) survives honestly → present it as the
  single qualified positive, everything else negative.
- Whether to attempt any DTM salvage for a true CHM, or accept the DSM caveat and move on.
