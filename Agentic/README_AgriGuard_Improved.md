# AgriGuard-ML

## Explainable Apple Leaf Disease Detection with a Proposed Agentic Safety Layer

> **Project status:** the classical ML disease-classification baseline exists. The agentic validation, visible-lesion severity measurement, retrieval-augmented guidance, and LLM advisory described below are **proposed extensions** until their code, labelled datasets, and evaluation reports are added.

AgriGuard-ML is an Apple-only disease-classification project built on handcrafted image features and classical machine learning. Its current baseline uses a 848-dimensional feature vector and selects **LightGBM** as the deployment candidate because it performed best in the recorded model comparison. The project is being extended with a staged, testable safety layer that will reject unsuitable inputs before diagnosis and, later, provide evidence-grounded farmer guidance.

---

## Table of Contents

1. [Current Scope and Status](#current-scope-and-status)
2. [Supported Classes](#supported-classes)
3. [Implemented Baseline](#implemented-baseline)
4. [Current Architecture](#current-architecture)
5. [Proposed Agentic Extension](#proposed-agentic-extension)
6. [Minimal First Agentic Workflow](#minimal-first-agentic-workflow)
7. [Shared Input and Output Contract](#shared-input-and-output-contract)
8. [Required Data for Proposed Agents](#required-data-for-proposed-agents)
9. [Evaluation Plan](#evaluation-plan)
10. [Configuration and Calibration](#configuration-and-calibration)
11. [Safety and Hallucination Controls](#safety-and-hallucination-controls)
12. [Milestone Plan](#milestone-plan)
13. [Repository Documentation Rules](#repository-documentation-rules)
14. [Limitations](#limitations)

---

## Current Scope and Status

| Capability | Status | Evidence required before status can change |
|---|---|---|
| Apple disease classification using handcrafted features | Implemented baseline | Reproducible training and evaluation artifacts |
| Feature extraction with 848 features | Implemented baseline | Versioned extraction code and feature checks |
| KNN, SVM, Random Forest, and LightGBM comparison | Implemented baseline | Recorded comparison report |
| LightGBM selection as current predictor | Implemented baseline | Recorded comparison shows it is the leading candidate |
| Soft-voting or stacking deployment | Not selected for current baseline | Must outperform LightGBM on validation under a predefined selection metric |
| Image quality gate | Proposed | Labelled quality dataset, calibration, code, and tests |
| Leaf presence gate | Proposed | Labelled leaf/non-leaf dataset, code, and tests |
| Apple-leaf identity validator | Proposed | Labelled Apple/non-Apple leaf dataset, code, and tests |
| Visible-lesion severity estimator | Proposed | Pixel-level leaf and lesion masks plus severity-error evaluation |
| Retrieval-augmented guidance | Proposed | Curated source corpus, retrieval evaluation, and citation trace |
| LLM advisory | Proposed | Grounded-output safety tests and agriculture-domain review |

The project must not claim that any proposed component is operational until the required artifacts exist.

---

## Supported Classes

The current verified disease dataset uses the PlantVillage Apple class folders:

1. `Apple___Apple_scab`
2. `Apple___Black_rot`
3. `Apple___Cedar_apple_rust`
4. `Apple___healthy`

This README is intentionally **Apple-only**. Adding other crops or diseases requires a separately documented dataset, label map, train/validation/test split, evaluation report, and model-selection decision.

---

## Implemented Baseline

### Purpose

Given an image belonging to one of the supported Apple classes, extract handcrafted visual features and predict one disease class using the selected classical ML model.

### Feature vector

The implemented feature pipeline produces **848 features**:

| Family | Dimensions | Purpose |
|---|---:|---|
| Color | 524 | RGB/HSV statistics and histograms describing visible colour distribution |
| Texture | 23 | Texture descriptors such as GLCM and LBP-derived measurements |
| Shape | 13 | Leaf or region geometry descriptors when segmentation is available |
| HOG | 288 | Gradient and edge-structure descriptors |
| **Total** | **848** | Concatenated feature vector |

The previous 54/24/7/704 breakdown is not the implementation source of truth and must not be used in project documents unless the extractor is changed and re-evaluated.

### Models evaluated

- K-Nearest Neighbours
- Support Vector Machine
- Random Forest
- LightGBM
- Optional ensembles for experimentation only

### Current model-selection decision

**LightGBM is the current baseline deployment candidate.** It is selected because it performed better than the recorded alternatives, including the evaluated ensembles. Ensembles remain research experiments; they are not described as the central diagnostic method unless a new, reproducible evaluation shows a clear improvement under the pre-defined selection metric.

### Leakage control

For every experiment:

1. Split data before learning transformations.
2. Fit the scaler on the training partition only.
3. Fit PCA or feature selection on the training partition only, if used.
4. Apply the fitted transformations to validation and test data without refitting.
5. Reserve the final test split for final reporting only.

---

## Current Architecture

```text
Apple Leaf Image
      |
      v
Preprocessing and optional leaf segmentation
      |
      v
Handcrafted feature extraction
Color 524 + Texture 23 + Shape 13 + HOG 288 = 848 features
      |
      v
Training-fitted scaler and optional selection transform
      |
      v
LightGBM baseline predictor
      |
      v
One of four supported Apple classes plus calibrated confidence
```

The baseline classifier operates on its supported label space. It must not be presented as a general crop recognizer, a non-leaf detector, or a treatment recommender.

---

## Proposed Agentic Extension

The proposed agentic layer is a **hybrid** design. Deterministic computer vision and trained ML perform measurements and classification. A language model, if added, is restricted to explaining evidence and summarizing retrieved guidance.

```text
Uploaded Image
      |
      v
A1 Image Quality Gate                 Proposed
      |
      v
A2 Leaf Presence Gate                 Proposed
      |
      v
A3 Apple-Leaf Identity Validator      Proposed
      |
      v
A4 Policy Controller                  Proposed
      |
      +--> Reject, retake request, or human-review request
      |
      v
Implemented LightGBM Apple disease baseline
      |
      +-------------------------+
      v                         v
A5 Visible-lesion estimator     A6 Knowledge retrieval agent
Proposed                        Proposed
      |                         |
      +-----------+-------------+
                  v
A7 Grounded advisory generator
Proposed
```

### Why this separation matters

- OpenCV and segmentation measure pixels, masks, brightness, blur, and visible area.
- The LightGBM baseline classifies within the known Apple disease label space.
- Retrieval provides source-bound agricultural guidance.
- The LLM may explain structured results but may not change numeric measurements, invent a diagnosis, create citations, or fabricate chemical rates.

---

## Minimal First Agentic Workflow

Do **not** begin with an LLM, RAG, severity scoring, or a full multi-agent interface. First implement and validate the smallest useful workflow:

```text
Upload image
  -> deterministic quality checks
  -> deterministic leaf-presence check
  -> explicit policy decision
  -> existing LightGBM predictor only if accepted
  -> structured result or clear rejection reason
```

### Minimal agents

| Agent | Role | Status |
|---|---|---|
| A1 Quality Agent | Decode, resolution, blur, brightness, and glare checks | Proposed first implementation |
| A2 Leaf Presence Agent | Leaf-like foreground evidence using configurable CV rules | Proposed first implementation |
| A3 Policy Agent | PASS, RETAKE, REJECT, or REVIEW decision | Proposed first implementation |
| Baseline Predictor | Existing LightGBM model for accepted Apple images | Implemented baseline |

### Minimal acceptance criteria

Before adding other agents:

- Every result follows the schema below.
- Invalid files, blurred images, dark images, non-leaf images, and valid Apple images have dedicated tests.
- Rejected inputs never reach the disease classifier.
- Valid inputs produce a disease label and calibrated confidence.
- The quality and leaf thresholds are calibrated on held-out labelled validation data.
- The false-rejection rate for valid Apple images is reported.

---

## Shared Input and Output Contract

### Input

```json
{
  "image_id": "upload-001",
  "image_path": "path-or-storage-reference",
  "region": null,
  "language": "en"
}
```

### Minimal validation output

```json
{
  "image_id": "upload-001",
  "policy": {
    "decision": "PASS",
    "reason_code": "VALID_INPUT",
    "message": "Image quality and leaf presence checks passed."
  },
  "quality": {
    "passed": true,
    "blur_score": 146.8,
    "brightness_score": 132.5,
    "resolution": [1024, 768],
    "threshold_profile": "validation-profile-v1"
  },
  "leaf_presence": {
    "passed": true,
    "leaf_area_fraction": 0.58,
    "rule_profile": "validation-profile-v1"
  },
  "diagnosis": {
    "attempted": true,
    "model": "lightgbm-baseline",
    "model_version": "recorded-model-version",
    "label": "Apple___Apple_scab",
    "confidence": 0.94
  }
}
```

### Failure behaviour

| Policy decision | Meaning | Disease model runs? | Required user message |
|---|---|---:|---|
| `RETAKE` | Image cannot be reliably assessed | No | Explain the specific camera or framing correction |
| `REJECT` | Input is outside the supported use case | No | Explain that the current system supports Apple leaves only |
| `REVIEW` | Evidence is ambiguous | No by default | Ask for another photo or professional review |
| `PASS` | Validation checks pass | Yes | Return the baseline diagnosis and calibrated confidence |

---

## Required Data for Proposed Agents

The disease-classification dataset alone is insufficient for validation and severity agents.

### A1 Quality Agent dataset

Create a labelled image-quality dataset containing:

- sharp valid Apple leaf images
- controlled blurred variants and real blurred field photos
- underexposed images
- overexposed or glare-affected images
- low-resolution and improperly framed images

For each image, record a human quality label, failure reason, blur score, brightness measurement, and whether a disease diagnosis would be trustworthy.

**Primary metrics:** rejection precision, rejection recall, false-rejection rate, and calibration stability across devices and lighting conditions.

### A2 Leaf Presence dataset

Create a labelled image set containing:

- valid Apple leaves
- other plant leaves
- fruit, bark, soil, tools, hands, and blank backgrounds
- leaf images with cluttered backgrounds

Annotate leaf-present versus leaf-absent and, for a representative subset, leaf bounding boxes or masks.

**Primary metrics:** leaf-presence accuracy, precision, recall, non-leaf rejection rate, and false rejection of valid Apple leaves.

### Apple-leaf identity dataset

Build a balanced binary dataset of Apple versus non-Apple leaves. Non-Apple examples should include visually similar crop and tree leaves, not just obvious non-leaf objects.

**Primary metrics:** binary macro-F1, false acceptance of non-Apple leaves, false rejection of Apple leaves, probability calibration, and out-of-distribution behaviour.

### Visible-lesion severity dataset

Visible-lesion percentage cannot be claimed without reference masks. Collect a separate annotated subset where trained annotators provide:

- leaf mask
- lesion mask
- annotation guidelines for lesion boundaries
- image-quality label
- disease label

The output is a **visible lesion area estimate**, not an exact measure of biological infection, field prevalence, or expected yield loss.

**Primary metrics:** leaf-mask IoU, lesion-mask IoU, Dice score, mean absolute percentage error, and severity-tier agreement.

---

## Evaluation Plan

| Component | Required metrics | Evaluation dataset |
|---|---|---|
| LightGBM disease baseline | accuracy, macro-F1, per-class precision/recall, confusion matrix, calibration, inference time | held-out Apple disease test split |
| A1 Quality Agent | precision, recall, false-rejection rate | labelled image-quality split |
| A2 Leaf Presence Agent | precision, recall, accuracy, false-rejection rate | leaf/non-leaf validation split |
| Apple identity validator | macro-F1, Apple false rejection, non-Apple false acceptance | balanced Apple/non-Apple split |
| Severity estimator | leaf IoU, lesion IoU, Dice, MAE of visible-lesion percent, tier agreement | pixel-mask validation split |
| Retrieval agent | source precision, citation completeness, region match, freshness | curated question and source set |
| Advisory agent | JSON validity, citation-grounding rate, unsupported-claim rate, expert review score | safety and agriculture-domain review set |
| Full pipeline | accepted-case disease macro-F1, invalid-input rejection, false rejection, latency, cost | combined held-out end-to-end set |

The core comparison is:

```text
Disease baseline without validation
versus
Validation gates + disease baseline on accepted inputs
```

Report both accepted-case performance and false-rejection behaviour; an apparent accuracy gain is not useful if valid Apple images are rejected too often.

---

## Configuration and Calibration

No fixed threshold in this README is a universal production rule.

Blur, brightness, leaf-area, Apple-identity confidence, segmentation-quality, and severity-tier thresholds must be stored in a versioned configuration file and calibrated against labelled validation data. Every saved result should record the configuration profile and model version used.

Example configuration structure:

```yaml
profile: validation-profile-v1
quality:
  min_blur_score: calibrated-on-quality-val-set
  brightness_range: calibrated-on-quality-val-set
leaf_presence:
  min_leaf_area_fraction: calibrated-on-leaf-val-set
apple_identity:
  pass_confidence: calibrated-on-apple-identity-val-set
policy:
  low_confidence_action: review
severity:
  tier_boundaries: validated-per-supported-disease
```

---

## Safety and Hallucination Controls

### Deterministic evidence boundaries

- Blur, brightness, leaf masks, lesion masks, and visible lesion percentage are produced by CV/ML code, not by an LLM.
- The LLM receives these values as read-only evidence.
- The advisory layer may never alter a calculated percentage or fabricate a confidence score.

### Diagnostic boundaries

- The baseline is restricted to its four supported PlantVillage Apple classes.
- An image outside the supported scope must be rejected or routed for review.
- Low-confidence or ambiguous cases must not receive a definitive disease diagnosis.

### Retrieval boundaries

- Retrieval sources must have organization, publication date, crop, disease, region, and source URL metadata.
- The retrieval system must return insufficient-evidence status instead of guessing.
- Agricultural advice must cite retrieved passages.

### Chemical advice boundaries

- Do not hard-code pesticide products, dilution rates, pre-harvest intervals, or regulatory claims in the model prompt or README.
- Any chemical guidance must be backed by a retrieved, current, region-appropriate authoritative source and conform to the registered product label.
- If region or evidence is missing, the system may give general cultural precautions and should recommend a local extension officer or licensed agronomist.
- Severity alone must never determine a pesticide recommendation.

### LLM role

A Gemini Flash-class model can be evaluated later for multilingual, structured advisory output. It is not part of the current baseline and it must have a deterministic template-based fallback for outages or unsupported conditions.

---

## Milestone Plan

### Milestone 0: Documentation and baseline reconciliation

**Goal:** Establish a single source of truth.

**Deliverables**
- README reflects 848 features as 524/23/13/288.
- LightGBM is documented as the baseline selection.
- Existing results are linked or summarised with model and data versions.
- `README.md`, `ai_agent.md`, `development.md`, and other project documents label each capability as implemented or proposed.

**Acceptance criteria**
- No document states a different feature breakdown without a corresponding implementation change.
- No document presents an unimplemented agent as deployed.

### Milestone 1: Minimal validation gate

**Goal:** Implement A1, A2, policy control, and existing LightGBM prediction.

**Dependencies**
- Labelled quality and leaf-presence datasets.
- Versioned threshold configuration.

**Acceptance criteria**
- Valid and invalid paths are covered by unit and integration tests.
- The pipeline short-circuits correctly.
- False-rejection rate is reported.

### Milestone 2: Apple-leaf identity validation

**Goal:** Add a calibrated Apple versus non-Apple binary model.

**Dependencies**
- Balanced binary dataset and held-out test split.

**Acceptance criteria**
- Binary macro-F1, Apple false rejection, and non-Apple false acceptance are reported.
- Marginal cases route to retake or review rather than the disease model.

### Milestone 3: Visible-lesion severity estimator

**Goal:** Estimate visible lesion area with annotated mask validation.

**Dependencies**
- Reference leaf and lesion masks.

**Acceptance criteria**
- Mask quality, percentage error, and tier agreement meet pre-defined thresholds.
- UI labels the result as a visible-lesion estimate.

### Milestone 4: Grounded retrieval and advisory

**Goal:** Add source-bound farmer guidance.

**Dependencies**
- Curated and metadata-filtered agriculture corpus.
- Region and language input.
- Domain-expert safety review.

**Acceptance criteria**
- Every guidance item has a source trace.
- Unsupported chemical claims are blocked.
- Missing evidence results in abstention or human-review guidance.

### Milestone 5: End-to-end evaluation and interface

**Goal:** Deliver a testable Streamlit workflow.

**Acceptance criteria**
- Image upload, validation trace, diagnosis, overlay, and review/retake paths work.
- Full-pipeline metrics, latency, and audit traces are reproducible.

---

## Repository Documentation Rules

To prevent future contradictions:

1. This README is the high-level source of truth for scope and implementation status.
2. `docs/implementation_status.md` must list each component, code path, model artifact, test artifact, and report.
3. `docs/data_contracts.md` must define dataset schema, labels, sources, and annotation requirements.
4. `docs/evaluation_protocol.md` must define splits, metrics, thresholds, and model-selection rules.
5. Experiments must record data version, feature-pipeline version, configuration profile, random seed, model version, and metrics.
6. A planned feature must remain labelled **Proposed** until its code and evaluation artifacts are available.

---

## Limitations

- PlantVillage images may not represent field conditions such as complex backgrounds, shadows, mixed symptoms, or low-cost camera quality.
- The baseline only supports four Apple disease labels and cannot reliably classify arbitrary crops or diseases.
- Visible lesion area is not equivalent to whole-tree infection, orchard-wide incidence, yield loss, or treatment urgency.
- Image segmentation can fail under poor lighting, water droplets, natural leaf aging, or severe occlusion.
- AgriGuard is decision support; it does not replace local agronomic inspection or compliance with pesticide labels and regulations.

---

## Next Recommended Action

Implement **Milestone 0** first: reconcile repository documentation with the real feature extractor, recorded model comparison, data labels, and executable commands. Then build and evaluate the minimal validation gate before adding severity, RAG, or an LLM.
