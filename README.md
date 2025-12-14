\# EPA CO₂ Architecture Model



This project builds a machine-learning–ready dataset to predict vehicle CO₂ emissions

for **early-phase powertrain and vehicle architecture decisions**.



The goal is to estimate CO₂ risk \*\*before\*\* expensive simulations or physical testing,

using historical EPA certification data.



\## Scope (Version 1)

\- Gasoline ICE vehicles only

\- Model years: 2010–2025

\- Vehicle-level aggregation using EPA FTP (city) and HWY (highway) cycles

\- CO₂ target computed using official 55% / 45% weighting



\## What this repository contains

\- `pipelines/` : Reproducible data processing scripts

\- `data/`      : Documentation about data sources (raw data not stored)

\- `artifacts/` : Generated datasets and outputs (ignored by Git)



\## Current status

\- Dataset v1 defined and versioned

\- Feature schema frozen

\- Ready for train/validation/test split and modeling



\## Notes

Raw EPA data files and generated artifacts are intentionally excluded from version control.

All datasets can be reproduced by running the pipeline scripts.



