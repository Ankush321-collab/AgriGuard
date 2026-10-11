# AgriGuard-ML

### An Agentic, ML-Only Framework for Apple Leaf Disease Detection

[![Status](https://img.shields.io/badge/status-in--development-yellow)]()
[![Approach](https://img.shields.io/badge/approach-classical%20ML%20%2B%20agentic%20validation-blue)]()
[![No Deep Learning](https://img.shields.io/badge/deep%20learning-none-lightgrey)]()

> A classical machine-learning disease classifier for apple leaves — KNN, SVM, LightGBM, Random Forest, and an ensemble on top — guarded by an agentic validation layer that checks image quality, leaf presence, and species identity *before* diagnosis ever runs.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Problem Statement & Motivation](#2-problem-statement--motivation)
3. [System Architecture](#3-system-architecture)
4. [Tech Stack](#4-tech-stack)
5. [Development Plan — Phase by Phase](#5-development-plan--phase-by-phase)
6. [Project Timeline Summary](#6-project-timeline-summary)
7. [Folder Structure](#7-folder-structure)
8. [Evaluation Strategy](#8-evaluation-strategy)
9. [Risks & Mitigations](#9-risks--mitigations)
10. [Getting Started](#10-getting-started)
11. [Roadmap / Future Work](#11-roadmap--future-work)

---

## 1. Overview

**AgriGuard-ML** detects diseases in apple leaves (e.g. Healthy, Scab, Rust, Multiple Diseases) using **only classical machine learning** — no deep learning, no GPU dependency. Predictions are made from handcrafted visual features (color, texture, shape, gradient) rather than features learned by a neural network, which keeps the system fast, lightweight, and interpretable.

Sitting in front of the classifier is a small **agentic validation pipeline**: four cooperating agents that check whether an uploaded image is even usable — is it in focus, does it contain a leaf, is that leaf actually an apple leaf — before the disease model is allowed to run. Images that fail are rejected with a clear reason instead of being force-fit into a disease prediction.

The project is built in two clearly separated stages:

- **Stage 1 — ML Core**: built, trained, and proven completely on its own first.
- **Stage 2 — Agentic Gate**: built second, wrapped around the proven ML core.

This ordering is deliberate — see [Section 5](#5-development-plan--phase-by-phase) for why.

---

## 2. Problem Statement & Motivation

Design and implement a robust, explainable, ML-only apple leaf disease detection system that:

1. Accepts an uploaded image.
2. Runs it through a multi-agent validation pipeline that verifies image quality, leaf presence, and apple-leaf identity.
3. Only proceeds to disease classification once validation passes.
4. Classifies disease using handcrafted visual descriptors and classical ML models (KNN, SVM, LightGBM, Random Forest), with an optional ensemble fusion layer.
5. Returns a reliable prediction with a confidence score — or, on validation failure, a clear reason and actionable guidance.

**Gaps this project closes, compared to typical systems:**

| Gap | Typical system | AgriGuard-ML |
|---|---|---|
| Input trustworthiness | Diagnoses whatever is uploaded | Validates image before diagnosis |
| Model governance | Black-box CNN | Interpretable handcrafted features + classical ML |
| Pipeline robustness | No guardrails for bad input | Explicit multi-agent checks with reasons |
| Compute footprint | GPU-dependent | CPU-friendly, lightweight |

---

## 3. System Architecture

```
                     ┌──────────────────────────────┐
                     │      STAGE 2 — Agentic Gate    │
                     │        (built second)          │
                     │                                 │
   image ──────────▶ │  Quality Agent (A1)             │
                     │        │ pass                   │
                     │        ▼                        │
                     │  Leaf Presence Agent (A2)        │
                     │        │ pass                   │
                     │        ▼                        │
                     │  Apple-Leaf Validator Agent (A3) │
                     │        │ pass                   │
                     │        ▼                        │
                     │  Policy Agent (A4) ──── fail ───▶│──▶ rejection + reason
                     └────────┼───────────────────────┘
                              │ pass
                              ▼
                     ┌──────────────────────────────┐
                     │      STAGE 1 — ML Core         │
                     │      (built first, proven      │
                     │       standalone)               │
                     │                                 │
                     │  preprocess → segment (opt.)    │
                     │  → feature extraction           │
                     │    (color + GLCM + LBP +        │
                     │     shape + HOG)                │
                     │  → scaler / PCA                 │
                     │  → KNN / SVM / LGBM / RF         │
                     │  → ensemble (voting/stacking)    │
                     └──────────────────────────────┘
                              │
                              ▼
                  disease class + confidence + model used
```

---

## 4. Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Classical ML | scikit-learn, LightGBM |
| Feature extraction | OpenCV, scikit-image (GLCM, LBP, HOG) |
| Data handling | pandas, NumPy |
| App / UI | Streamlit |
| Testing | pytest |
| Experiment tracking | CSV/JSON reports (`reports/`), optionally MLflow |
| Packaging | `requirements.txt`, virtualenv/conda |

No TensorFlow, PyTorch, or GPU dependency at any stage.

---

## 5. Development Plan — Phase by Phase

The build order is intentional: **Phases 0–7 build and prove the ML core in isolation. Phases 8–12 build the agentic layer around that proven core.** This means if something breaks after Phase 8, you know immediately it's the agentic layer, not the model — debugging stays cheap.

Each phase below lists **objective, tasks, deliverables, and a verification checklist**. Do not start the next phase until the current phase's checklist is fully checked off.

---

### PHASE 0 — Planning & Scope Freeze
**Objective:** Lock every definition the rest of the project will build against.

**Tasks**
- Freeze disease classes (e.g. Healthy, Scab, Rust, Multiple Diseases).
- Freeze rejection classes for the agentic layer (non-leaf, non-apple-leaf, poor quality).
- Define minimum image-quality thresholds (blur, brightness, resolution).
- Define target metrics: disease macro-F1, validation accuracy, end-to-end accepted-image accuracy, false-rejection rate.

**Deliverables**
- `docs/scope.md`
- `docs/acceptance_criteria.md`

**Verification Checklist**
- [ ] Every disease class has a one-line definition with example images identified.
- [ ] Acceptance criteria state numeric pass/fail thresholds for every metric listed above.
- [ ] No ambiguity remains about "Healthy" vs. borderline/early-stage disease.

---

### PHASE 1 — Repository Refactor (ML-Only Baseline)
**Objective:** Remove any legacy deep-learning path and establish a clean, modular repo.

**Tasks**
- Preserve any existing CNN-based app as `app_cnn_legacy.py` (backup only).
- Create the target structure: `src/`, `models/`, `configs/`, `reports/`, `tests/`, `docs/`.
- Remove TensorFlow / EfficientNet / PyTorch dependencies.
- Add the ML-only dependency set to `requirements.txt`.

**Deliverables**
- Clean, importable `src/` package with zero DL imports.

**Verification Checklist**
- [ ] `grep -ri "tensorflow\|keras\|torch" src/` returns nothing.
- [ ] `app_cnn_legacy.py` still runs standalone as a reference.
- [ ] All directories exist and imports resolve without error.

---

### PHASE 2 — Data Foundation
**Objective:** Build the dataset manifest and splits everything downstream depends on.

**Tasks**
- Build a manifest per image: path, disease label, source, and validation tags (apple-leaf / non-apple-leaf / invalid-quality) — collect the validation tags now even though they're only used from Phase 8 onward.
- Create stratified train/val/test splits.
- Set aside a separate small dataset of non-leaf and non-apple-leaf images for later agent training/evaluation.

**Deliverables**
- `data/interim/splits/{train,val,test}.csv`
- Class-distribution report

**Verification Checklist**
- [ ] Splits are non-overlapping and stratified — every class appears in every split.
- [ ] Class distribution report flags any severely underrepresented class.
- [ ] Manifest schema is documented (column names, types, meaning).

---

### PHASE 3 — Preprocessing & Segmentation
**Objective:** Standardize every image before feature extraction.

**Tasks**
- Resize all images to a fixed resolution (e.g. 256×256).
- Convert to RGB / HSV / grayscale as needed by downstream feature extractors.
- Optional: segment the leaf from the background (HSV threshold + morphology + largest contour).
- Handle edge cases: tiny masks, cluttered backgrounds, multiple leaves per image.

**Deliverables**
- `src/preprocess.py`, `src/segment.py`

**Verification Checklist**
- [ ] Every manifest image survives preprocessing without crashing.
- [ ] Manual spot-check of 20–30 segmented images shows reasonable leaf outlines.
- [ ] Edge cases degrade gracefully (no hard crash on noisy/atypical input).

---

### PHASE 4 — Feature Engineering
**Objective:** Turn each preprocessed image into a stable, explainable feature vector.

**Feature families**
- **Color** — RGB/HSV mean & std, color histograms
- **Texture** — GLCM (contrast, energy, homogeneity, correlation), LBP histogram
- **Shape** — area, perimeter, circularity (where segmented)
- **Gradient** — HOG descriptors

All families are concatenated into a single per-image feature vector.

**Deliverables**
- `src/features.py`, `src/feature_pipeline.py`
- Logged feature-vector length and version

**Verification Checklist**
- [ ] Feature vector length is fixed and logged per pipeline version.
- [ ] No NaN/Inf values across the full dataset.
- [ ] Average feature-extraction time per image is logged and acceptable.

---

### PHASE 5 — Model Training (Disease Classifiers)
**Objective:** Train the four classical models on the handcrafted features.

**Tasks**
- Fit `StandardScaler` on **train split only**.
- Optional PCA, also fit on train only.
- Hyperparameter tuning via `RandomizedSearchCV` for each of: KNN, SVM (RBF, `probability=True`), LightGBM, Random Forest.
- Save each best model variant.

**Deliverables**
- `models/knn.pkl`, `models/svm.pkl`, `models/lgbm.pkl`, `models/rf.pkl`
- `models/scaler.pkl`, `models/label_encoder.pkl`, optional `models/pca.pkl`

**Verification Checklist**
- [ ] All four model files load without error.
- [ ] Scaler/label-encoder (and PCA, if used) saved alongside.
- [ ] Each model's val-set macro-F1 beats a majority-class baseline.
- [ ] Confirmed the scaler/PCA `.fit()` call only ever saw the training split (no leakage).

---

### PHASE 6 — Ensemble Layer
**Objective:** Combine the four base models into one stronger, still-ML-only predictor.

**Tasks**
- Soft voting across KNN / SVM / LightGBM / RF.
- Stacking with a Logistic Regression meta-model.
- Compare both against the best single model.

**Deliverables**
- `models/voting.pkl`, `models/stacking.pkl`
- `best_model_selection.json`

**Verification Checklist**
- [ ] Both ensemble variants evaluated on the val split.
- [ ] Selection file records which model/ensemble won, with the metric that decided it.
- [ ] If the ensemble does *not* beat the best single model, that's documented honestly rather than papered over.

---

### PHASE 7 — ML Core Evaluation (Disease Classifier Only)
**Objective:** Prove Stage 1 works completely on its own, before any agentic wrapping.

**Metrics (on the test set)**
- Accuracy, macro-F1, per-class precision/recall, confusion matrix, inference time.

**Deliverables**
- `reports/disease_metrics.csv`, confusion-matrix visualization

**Verification Checklist**
- [ ] All metrics meet the thresholds frozen in Phase 0.
- [ ] Confusion matrix reviewed for systematic class confusions.
- [ ] This baseline is saved and will be used again in Phase 9 for the with/without-agents comparison.

**✅ Stage 1 (ML Core) is now complete and provable in isolation. Proceed to the agentic layer.**

---

### PHASE 8 — Agentic Validation Pipeline
**Objective:** Build the four-agent gatekeeper that sits in front of the disease classifier.

**Agent A1 — Quality Agent**
- Checks blur (Laplacian variance), illumination, minimum resolution.
- Output: `quality_pass: bool`, `quality_reason`.

**Agent A2 — Leaf Presence Agent**
- Green-mask + contour heuristics, plus basic texture/shape sanity checks.
- Output: `leaf_present: bool`, `leaf_bbox`, `mask_quality`.

**Agent A3 — Apple-Leaf Validator Agent**
- Small **binary** ML classifier (apple leaf vs. not) reusing the Phase 4 feature pipeline; trained the same way as Phase 5 (SVM/RF baseline, best selected).
- Output: `is_apple_leaf: bool`, `confidence`.

**Agent A4 — Policy Agent**
- Rule-based decision: A1 fail → reject · else A2 fail → reject · else A3 confidence below threshold → reject/flag · else → forward to Stage 1.

**Deliverables**
- `src/agents/quality_agent.py`
- `src/agents/leaf_presence_agent.py`
- `src/agents/apple_leaf_agent.py`
- `src/agents/policy_agent.py`
- `src/agents/orchestrator.py`

**Verification Checklist**
- [ ] Each agent is independently unit-testable with known-good and known-bad inputs.
- [ ] `orchestrator.py` chains A1 → A2 → A3 → A4 correctly and short-circuits on first failure.
- [ ] A3 hits an acceptable binary F1 on the held-out non-leaf/non-apple-leaf set from Phase 2.
- [ ] Every rejection returns a human-readable reason, not just a boolean.

---

### PHASE 9 — Two-Level Evaluation
**Objective:** Measure the validation agents and the disease classifier both separately and together.

**Level 1 — Validation agents**
- Quality-pass precision/recall, leaf-presence accuracy, apple-leaf binary F1.

**Level 2 — Disease classifier on agent-accepted images**
- Same metrics as Phase 7, but computed only on the "gated" subset.

**End-to-end pipeline metrics**
- Overall rejection rate, false-rejection rate, accepted-case disease accuracy.

**Deliverables**
- `reports/validation_metrics.csv`
- `reports/disease_metrics_gated.csv`

**Verification Checklist**
- [ ] Phase 7 (ungated) vs. Phase 9 (gated) accuracy are directly compared — this is the evidence the agentic layer adds value.
- [ ] False-rejection rate is within the range set in Phase 0.
- [ ] Both report files exist and are reproducible from a single script/command.

---

### PHASE 10 — Streamlit App (Full Pipeline)
**Objective:** Ship the end-to-end user experience.

**UI flow**
1. User uploads an image.
2. Validation agents run in sequence; a visible checklist shows pass/fail + reason per agent.
3. Pass → disease inference runs; shows predicted class, confidence, model/ensemble used.
4. Fail → actionable guidance ("try a closer shot," "improve lighting," "this doesn't look like an apple leaf").

**Deliverables**
- `app.py` with two tabs: **Validation Trace** and **Disease Prediction**.

**Verification Checklist**
- [ ] A deliberately bad image (blurry / non-leaf / wrong species) is rejected with a clear on-screen reason.
- [ ] A good image flows through both stages to a final prediction with confidence.
- [ ] App runs end-to-end with no DL dependency installed.

---

### PHASE 11 — Testing & QA
**Objective:** Confirm the system holds up outside the happy path.

**Test categories**
- **Unit** — feature-extractor output shape/ranges; each agent's decision on edge cases.
- **Integration** — full pipeline, upload → output, for both accept and reject paths.
- **Robustness** — blur/noise/low-light stress tests; non-leaf and wrong-species rejection checks.
- **Performance** — average inference time; memory usage under load.

**Deliverables**
- `tests/` suite, QA report

**Verification Checklist**
- [ ] `tests/` suite passes locally / in CI.
- [ ] QA report documents what was tested and what was fixed.
- [ ] Inference time meets the target stated in Phase 0 (e.g. "<2s per image on CPU").

---

### PHASE 12 — Documentation & Research Write-Up
**Objective:** Package the project so it's reproducible and explainable to someone else.

**Should cover**
- Motivation and research gap.
- Architecture diagram (Section 3 above).
- Agentic decision policy, spelled out clearly.
- ML feature-engineering details.
- Experimental setup and results — including the with-agents vs. without-agents ablation.
- Limitations and future work.

**Verification Checklist**
- [ ] README/report covers every point above.
- [ ] With/without-agents comparison is stated with numbers, not just described.
- [ ] Someone unfamiliar with the project could read the docs and understand why the agentic layer exists.

---

## 6. Project Timeline Summary

| Phase | Focus | Depends on |
|---|---|---|
| 0 | Scope & acceptance criteria | — |
| 1 | Repo refactor | 0 |
| 2 | Data foundation | 1 |
| 3 | Preprocessing & segmentation | 2 |
| 4 | Feature engineering | 3 |
| 5 | Model training | 4 |
| 6 | Ensemble layer | 5 |
| 7 | **ML core evaluation (milestone)** | 6 |
| 8 | Agentic validation pipeline | 7 |
| 9 | Two-level evaluation | 8 |
| 10 | Streamlit app | 9 |
| 11 | Testing & QA | 10 |
| 12 | Documentation & write-up | 11 |

Phases 0–7 are entirely ML. Phase 7 is the key milestone — a working, evaluated disease classifier with no agentic dependency. Phases 8–12 build, evaluate, and ship the agentic layer on top of that proven core.

---

## 7. Folder Structure

```
src/
  agents/
    quality_agent.py
    leaf_presence_agent.py
    apple_leaf_agent.py
    policy_agent.py
    orchestrator.py
  preprocess.py
  segment.py
  features.py
  feature_pipeline.py
  train.py
  ensemble.py
  evaluate.py
  predict.py
  validate_only.py
app.py
app_cnn_legacy.py
models/
  knn.pkl
  svm.pkl
  lgbm.pkl
  rf.pkl
  voting.pkl
  stacking.pkl
  scaler.pkl
  label_encoder.pkl
reports/
tests/
configs/
docs/
data/
  interim/
    splits/
requirements.txt
README.md
```

---

## 8. Evaluation Strategy

| What | Metric(s) | When |
|---|---|---|
| Disease classifier (ungated) | Accuracy, macro-F1, per-class P/R, confusion matrix, inference time | Phase 7 |
| Quality Agent | Precision/recall | Phase 9 |
| Leaf Presence Agent | Accuracy | Phase 9 |
| Apple-Leaf Validator Agent | Binary F1 | Phase 9 |
| Full pipeline | Rejection rate, false-rejection rate, gated accuracy | Phase 9 |
| App-level | Inference latency, memory usage | Phase 11 |

The single most important comparison in the whole project is **Phase 7 vs. Phase 9 gated accuracy** — it's the number that proves the agentic layer is worth the added complexity.

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Data leakage via scaler/PCA fit on val/test | Explicit checklist item in Phase 5; fit only on train split |
| Class imbalance | Stratified splits (Phase 2) + macro-F1 as primary metric, not raw accuracy |
| Agentic layer over-rejects valid images | False-rejection rate tracked explicitly (Phase 9); threshold tunable in Policy Agent |
| Segmentation failing on cluttered backgrounds | Segmentation kept optional; features degrade gracefully without it (Phase 3) |
| Feature pipeline drift between disease model and Apple-Leaf Validator | Both reuse the same `feature_pipeline.py` (Phase 4 / Phase 8) |

---

## 10. Getting Started

```bash
# clone and set up environment
git clone <repo-url>
cd agriguard-ml
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# run the app once models are trained (Phase 5+ complete)
streamlit run app.py
```

> Training scripts, config files, and dataset download instructions will be documented in `docs/` as each phase is completed.

---

## 11. Roadmap / Future Work

- Extend the Apple-Leaf Validator Agent to a multi-species leaf classifier, generalizing the gatekeeper beyond apple leaves.
- Add active-learning support so borderline/low-confidence predictions can be routed for human review and fed back into training.
- Explore on-device / offline deployment for low-connectivity field use, given the CPU-only footprint of the ML core.
