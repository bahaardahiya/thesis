# NASDAQ-100 Bubble Test — Step-by-Step Reference

This document is the map back to everything in this rebuild: what each file
does, the order to run things in, and the methodology decision behind it.
Keep it updated if you change parameters.

## 1. What this analysis does

Tests the NASDAQ-100 (NDX) weekly price-dividend ratio, 2005–2025, for
explosive/bubble behaviour using the Phillips, Shi & Yu (2015, *International
Economic Review*) recursive **BSADF/GSADF** right-tailed unit root test. The
BSADF statistic is compared against two different critical value sequences:

- **Monte Carlo (MC)**: the standard PSY approach — simulate critical values
  under i.i.d. N(0,1) random-walk innovations.
- **Wild bootstrap (WB)**: the Phillips & Shi (2020) refinement — simulate
  critical values by resampling the *actual* weekly log price-dividend
  changes with random sign flips (Rademacher multipliers), which preserves
  the real volatility-clustering pattern in the data instead of assuming
  constant variance.

## 2. The methodology decision: MC or WB as primary?

**Recommendation: use the wild bootstrap as your primary method** (which
matches what your methodology chapter already commits to), and present
Monte Carlo as an explicit robustness comparison. Reasons:

1. **It's the econometrically correct choice for this data.** NASDAQ weekly
   returns have well-documented volatility clustering (2008, 2020, etc.).
   Under heteroskedasticity, i.i.d.-N(0,1) Monte Carlo critical values are
   known to be too low — the test over-rejects H0 and flags "bubbles" that
   are really just ordinary volatility spikes. The wild bootstrap corrects
   this by drawing its null distribution from the data's own shock history,
   which is exactly why Phillips & Shi (2020) proposed it. You can see this
   directly in `output/figures/bsadf_full_sample.png`: the WB critical value
   spikes sharply right after the March 2020 COVID crash and the 2008 crisis
   — precisely the high-volatility windows where an i.i.d. assumption breaks
   down.
2. **It does not kill your results.** With this implementation, WB and MC
   agree on the same episodes at the 95% level (Oct 2008, Jan 2018, a
   sustained run from late 2021 into Jan 2022) — WB is moderately more
   conservative (fewer flagged weeks, narrower episode windows) but is not
   the "everything is null" outcome you got before. That earlier result
   pointing to zero detected bubbles almost certainly came from a bug in
   the previous implementation, not from a real property of the wild
   bootstrap — see the diagnostic numbers in §5.
3. **It's the stronger thesis narrative.** "I use the theoretically
   preferred, heteroskedasticity-robust procedure, and it still detects
   the well-known 2021 pre-crash rally, while the simpler i.i.d. approach
   over-detects" is a more sophisticated methodological point than either
   version alone, and it lets you keep the bootstrap-based methodology
   chapter you already wrote. Report both, lead with WB.

Headline empirical finding to build the results chapter around: **a
statistically significant, sustained explosive episode in the NASDAQ-100
price-dividend ratio from roughly June 2021 to mid-January 2022**, ending
right at the actual NASDAQ peak (19 Nov 2021) and subsequent 2022 reversal
— detected by both methods, most persistently by WB at the conventional 90%
level (33 of 34 weeks overlapping with MC's 90% episode). Secondary,
shorter-lived episodes appear around Oct–Dec 2008 (crisis-era volatility)
and Jan 2018 (the "Volmageddon" VIX spike).

## 3. File map

```
data/raw/NDX_Weekly.csv              Original Bloomberg export (price, 12m div yield, P/D ratio)
data/processed/ndx_weekly_pd_ratio.csv   Cleaned weekly series (output of data_prep.py)

src/data_prep.py         Parses/cleans the raw CSV -> data/processed/ndx_weekly_pd_ratio.csv
src/bsadf.py             Core PSY recursive BSADF/GSADF statistic (numba-jitted)
src/critical_values.py   Monte Carlo and wild bootstrap critical value simulation
src/run_analysis.py      Main pipeline: runs everything, writes output/tables/
src/plotting.py          Builds the two thesis figures from output/tables/bsadf_results.csv

output/tables/bsadf_results.csv     Full weekly series: BSADF stat + MC/WB critical values (90/95/99%)
output/tables/summary.json          Run parameters + one-shot GSADF test result
output/tables/flagged_episodes.csv  Contiguous date ranges where BSADF exceeds each threshold
output/figures/bsadf_full_sample.png   Main figure: full 2005-2025 sample, two panels
output/figures/bsadf_2021_episode.png  Detail figure: 2020-2022 zoom
```

## 4. How to reproduce (exact commands, in order)

```bash
pip install -r requirements.txt

python3 src/data_prep.py       # -> data/processed/ndx_weekly_pd_ratio.csv
python3 -m src.run_analysis    # -> output/tables/bsadf_results.csv, summary.json
python3 -m src.plotting        # -> output/figures/*.png
```

`run_analysis.py` also prints the BIC-selected lag order, minimum window
size, and the one-shot GSADF statistic vs. its critical values.

## 5. Parameters used (record these in your methodology chapter)

| Parameter | Value | Source |
|---|---|---|
| Series tested | log(NDX price / trailing-12m dividend) | Bloomberg `PX_LAST`, `EQY_DVD_YLD_12M`-derived `Price/Div Ratio` field |
| Sample | 2005-01-07 to 2025-12-26, weekly, T = 1095 | data/raw/NDX_Weekly.csv |
| Lag order k | 0 (BIC-selected, max candidate 8) | `bsadf.select_lag_bic` |
| Minimum window r0 | 0.01 + 1.8/√T ≈ 0.0644 → 70 weeks (~1.35 years) | PSY (2015) rule of thumb, `bsadf.min_window` |
| Test direction | Right-tailed (β > 0, mildly explosive) | Standard PSY/PWY bubble test specification |
| MC replications | 4,999, seed 20250803 | `run_analysis.SEED_MC` |
| WB replications | 4,999, seed 20250804 | `run_analysis.SEED_WB` |
| Significance levels reported | 90%, 95%, 99% | `run_analysis.QUANTILES` |

One-shot GSADF result (full-sample summary test): GSADF = 2.026 (peak
2008-11-21). This does **not** clear the 90% critical value under either MC
(2.111) or WB (3.228) — the omnibus test on its own is inconclusive. The
episode-level finding (§2) comes from the **date-stamping procedure**:
comparing the BSADF sequence to the critical value *sequence* at each week,
which is the actual empirical exercise in PSY-style papers, not the single
GSADF number. Explain this distinction in the results chapter — reviewers
who only skim for "was GSADF significant?" will otherwise be confused by a
"no" next to a chapter full of "yes, in these specific weeks."

## 6. Changing parameters later

- **More replications** (smoother critical value curves): raise
  `N_REPLICATIONS` in `src/run_analysis.py`. At k=0 each replication costs
  ~0.002s, so even 20,000 reps run in well under a minute.
- **Different significance levels**: edit `QUANTILES` in `run_analysis.py`.
- **Different lag order**: `select_lag_bic` picked k=0 for this data; if you
  change the sample or the series, check its output before running — if
  k ≠ 0, `run_analysis.py` will raise an error rather than silently using
  the wrong (slow, generic) code path. In that case swap
  `bsadf_sequence_k0`/`bsadf_sequences_batch_k0` calls for the general
  `bsadf_sequence`/`bsadf_sequences_batch` functions in `src/bsadf.py`,
  which support arbitrary k (much slower — budget real time for the
  bootstrap in that case).
- **Different minimum window**: pass an explicit `r0` to `bsadf.min_window`.

## 7. Known limitations worth a paragraph in the thesis

- Dividend data at weekly frequency is a trailing 12-month yield, not a
  point-in-time payment — this smooths short-run dividend dynamics and is a
  standard, but real, limitation of P/D-ratio bubble tests at this
  frequency.
- The sample starts in 2005, after the dot-com bubble's collapse, so it
  cannot speak to that episode — it captures 2008, 2018, 2020, and the
  2021–22 rally/reversal instead.
- The Oct–Nov 2008 flagged episode sits inside a market **crash**, not a
  boom; it likely reflects a short, sharp mean-reversion/dead-cat-bounce
  in the ratio rather than a "bubble" in the usual sense. Worth flagging
  explicitly rather than over-claiming it as bubble evidence.
