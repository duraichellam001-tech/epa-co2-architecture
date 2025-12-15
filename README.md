# 🚗 EPA CO₂ Architecture Screening Model

## Overview

This project develops a **data-driven CO₂ architecture screening framework** for **early-phase vehicle and powertrain concept evaluation**, using historical **EPA certification data**.

The goal is to provide **fast, architecture-level CO₂ estimates** *before* expensive CAE simulations, detailed vehicle models, or physical prototypes are available.

This work is intentionally positioned as a **front-end decision support tool**, not a replacement for high-fidelity simulation.

---

## Motivation

In early vehicle development phases, engineers must make architecture decisions (engine size, mass targets, drivetrain layout, transmission type) with **limited information**.

Today, these decisions are often:

* Spreadsheet-driven, or
* Deferred until late-stage simulations are available

Both approaches are slow, costly, and limit rapid exploration.

This project explores a key question:

> **How much of vehicle CO₂ behavior is already constrained by early architecture choices alone?**

---

## Scope (Phase-1)

**Vehicle domain**

* Gasoline ICE vehicles only
* EPA-certified passenger vehicles

**Data**

* EPA certification data
* Model years **2010–2025**

**Cycles**

* FTP (City)
* HWY (Highway)
* Combined CO₂ using official **55% City / 45% Highway** weighting

**Granularity**

* Vehicle-architecture level (not component level)
* Aggregated per vehicle configuration

---

## Modeling Objective

**Input (early-phase design parameters):**

* Model Year
* Engine Displacement (L)
* Equivalent Test Weight (lbs)
* Transmission Type (MT / AT / CVT)
* Drive System (FWD / RWD / AWD)

**Output:**

* Estimated combined CO₂ emissions (g/mi)

The intent is **relative architecture comparison**, not absolute certification prediction.

## Data Splitting Strategy

To reflect **real-world usage**, a **temporal split** was used:

* **Training:** Model years **2010–2021**
* **Testing:** Unseen future vehicles from **2022–2025**

This avoids data leakage and ensures the model is evaluated on **future technology**, not memorized history.

---

## Models

Two models were intentionally used:

### 1️⃣ Linear Regression (Baseline)

* Interpretable
* Physics-aligned
* Used to validate dataset construction and feature relevance

### 2️⃣ Tree-Based Model (GBDT)

* Captures non-linear interactions
* Higher predictive accuracy
* Used as the primary screening model

Both models are exposed in the inference UI for comparison.

---

## Performance Summary

**Tree-Based Model (Test Set: 2022–2025)**

* **R² ≈ 0.86**
* **MAE ≈ 25 g/mi**
* **RMSE ≈ 35 g/mi**

This level of accuracy is sufficient for **relative architecture trade studies** during early development phases.

---

## Interpretability & Engineering Insights (Linear Model)

Despite being purely data-driven, the linear model learned **physically consistent relationships**.

### Key Learned Effects

**Engine Displacement**

* ≈ **+40 g/mi per +1.0 L**
* Confirms fuel flow scaling with engine size

**Vehicle Mass**

* ≈ **+2.6 g/mi per +100 lbs**
* Matches expected road-load and inertial penalties

**Model Year**

* ≈ **−2.6 g/mi per year**
* Captures technology improvement, regulatory pressure, and efficiency gains
* Used as a proxy for **technology maturity**, not leakage

---

### Architecture-Level Effects

**Transmission (relative)**

* CVT: **−31 g/mi**
* AT: **+11 g/mi**
* MT: **+21 g/mi**

Reflects:

* Operating point optimization (CVT)
* EPA shift schedules
* Real certification outcomes

**Drive System**

* AWD: **+9 g/mi**
* FWD: **−8 g/mi**
* RWD: ~neutral

Captures:

* Drivetrain losses
* Mass penalties
* Layout efficiency differences

---

## Key Modeling Conclusion

The model is **not classifying vehicles by category**.

It learns **continuous, interpretable trade-offs** across:

* Mass
* Displacement
* Transmission
* Drivetrain

This directly addresses a common concern with data-driven screening tools.

---

## Engineering Interpretation (CAE Perspective)

The learned relationships resemble a **first-order vehicle energy balance**:

* Displacement → fuel flow scaling
* Weight → road load & inertia
* Transmission → operating efficiency
* Drivetrain → parasitic losses

In effect:

> **The ML model rediscovered vehicle physics without being explicitly told the equations.**

This makes it a **credible early-phase decision aid**, not a black box.

---

## What This Model Is Good For

* Early architecture trade studies
* “What-if” analysis:

  * +200 kg mass
  * AT → CVT
  * FWD → AWD
* Pre-screening concepts before simulation investment

---

## What It Is *Not* (By Design)

* Cycle-accurate fuel modeling
* Component-level loss breakdown
* Calibration-sensitive predictions

These limitations are intentional — and motivate Phase-2.

---

## Phase-2 Roadmap

Next, this work will be extended into a **Physics + ML hybrid framework**:

* Explicit power & energy balance models
* Cycle-aware physics baseline
* ML-based residual learning
* Improved extrapolation & engineering trust

---

## Live Demo

A lightweight inference UI has been deployed to demonstrate how the Phase-1 model can be used as an **architecture screening tool**:

👉 **Hugging Face Demo:**
https://huggingface.co/spaces/DuraiHF/epa-co2-architecture-screening

---

## Final Note

Even before detailed simulations exist, **architecture decisions already constrain CO₂ outcomes**.

This project quantifies that constraint and sets the foundation for physics-guided ML in early vehicle design.
