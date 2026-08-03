# thesis

NASDAQ-100 bubble detection using the Phillips, Shi & Yu (2015) BSADF/GSADF
recursive right-tailed unit root test on the weekly log price-dividend
ratio, 2005-2025, with both Monte Carlo and wild bootstrap critical values.

See `docs/ANALYSIS_STEPS.md` for the full pipeline, exact parameters, methodology
rationale, and how to reproduce every result and figure.

Quick start:

```bash
pip install -r requirements.txt
python3 src/data_prep.py
python3 -m src.run_analysis
python3 -m src.plotting
```