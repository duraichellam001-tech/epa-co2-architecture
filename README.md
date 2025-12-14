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

















\######----------------------------#####



INITIAL INTERPRETATIONS FROM PURE ML MODEL



\######----------------------------#######





1️⃣ KEY TECHNICAL INSIGHTS (README – “Results \& Interpretation”)

Baseline Model Performance (Path-A)



A linear regression model was trained on architecture-level features only, using a temporal train/test split (future model years held out).



Performance:



Train R² ≈ 0.81



Test R² ≈ 0.76



Test MAE ≈ 33 g/mi



Test RMSE ≈ 45 g/mi



Key takeaway:



Even with minimal, early-phase design inputs, the model explains ~75% of CO₂ variance on unseen future vehicles.



This validates:



Dataset construction



Feature relevance



Absence of data leakage



Suitability for early architecture screening



Learned Relationships Are Physically Consistent



The linear model coefficients exhibit clear alignment with vehicle physics and powertrain engineering principles.



Engine Displacement



+40 g/mi CO₂ per +1.0 L displacement



Confirms fuel flow scaling with engine size



Strong primary driver of CO₂ in certification data



Vehicle Mass



~+2.6 g/mi per +100 lbs



Matches expected road-load and inertial penalties



Effect magnitude consistent with EPA trends



Model Year



−2.6 g/mi per year



Captures:



Powertrain efficiency improvements



Regulatory tightening



Technology evolution (downsizing, better transmissions)



This validates inclusion of Model Year as a proxy for technology maturity, not leakage.



Architecture-Level Effects Are Quantified

Transmission Effects (relative)



CVT: −31 g/mi (most efficient)



AT: +11 g/mi



MT: +21 g/mi



These effects align with:



Engine operating point optimization (CVT)



EPA shift schedules



Real certification outcomes (manual ≠ always efficient)



Drive System Effects



AWD: +9 g/mi



FWD: −8 g/mi



RWD: ~neutral



The model clearly captures:



Additional drivetrain losses



Mass penalties



Layout efficiency differences



Important Modeling Conclusion



The model is not classifying vehicles by type.

It is learning continuous, interpretable trade-offs across mass, displacement, transmission, and drivetrain choices.



This directly addresses a common concern with data-driven screening models.



2️⃣ ENGINEERING-LEVEL INTERPRETATION (CAE-Focused)



This model behaves like a first-order vehicle energy balance, learned from data:



Displacement → fuel flow scaling



Weight → road load \& inertia



Transmission → operating efficiency



Drive layout → parasitic losses



In other words:



The ML model rediscovered vehicle physics without being explicitly told the equations.



This makes it a credible early-phase decision tool, not a black box.



What This Model Is Good For



Early architecture trade studies



“What-if” analysis:



+200 kg mass



AT → CVT



FWD → AWD



Pre-screening concepts before simulation investment



What It Is NOT (by design)



Cycle-accurate fuel modeling



Component-level loss breakdown



Calibration-sensitive predictions



These limitations are intentional — and motivate Path-B.



3️⃣ LINKEDIN STORYTELLING VERSION (High-Impact, Non-Academic)



You can reuse this almost verbatim 👇



🚗 Can We Predict Vehicle CO₂ Early — Without CFD or AMESim?



I built a production-style ML pipeline using EPA certification data (2010–2025) to explore whether early-phase vehicle architecture choices already encode enough signal to estimate CO₂ emissions.



Instead of focusing on brand or model names, I reduced vehicles to architecture-level features:



Engine displacement



Test weight



Transmission type



Drive layout



Model year (technology proxy)



A simple linear model trained on past years achieved:



~76% R² on future vehicles



~33 g/mi MAE — using only early design inputs



More interestingly, the model learned physically meaningful relationships:



Larger engines → higher CO₂



Heavier vehicles → higher CO₂



CVTs outperform ATs and MTs



AWD introduces a clear efficiency penalty



Newer vehicles show systematic CO₂ reductions



This confirmed something important:



Even before detailed simulations, architecture decisions already constrain CO₂ outcomes.



