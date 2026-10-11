# AgriGuard-ML

## An ML-Only, Agentic Apple Leaf Disease Detection System

AgriGuard-ML is a lightweight, explainable apple leaf disease detection project. It uses classical machine learning models instead of deep learning and adds a multi-agent validation and advisory layer around the disease classifier.

The system is designed for the **AppleLeaf9 dataset**, which contains nine apple leaf categories. Before predicting disease, the uploaded image is checked by validation agents. If the image is valid, handcrafted features are extracted and passed to ML models. After prediction, advisory agents estimate disease severity, explain symptoms, and suggest possible solutions.

---

# 1. Project Idea

In this project, I will build an ML-only apple leaf disease detection system for nine AppleLeaf9 categories. The system will first validate the uploaded image using multiple agents:

- **Quality Agent** checks blur, lighting, and resolution.
- **Leaf Presence Agent** confirms that a leaf is present.
- **Apple-Leaf Validator Agent** confirms that the leaf belongs to apple.
- **Policy Agent** decides whether the image should be accepted or rejected.

If the image passes validation, handcrafted features such as color, texture, shape, and HOG descriptors are extracted. These features are classified using classical ML models:

- KNN
- SVM
- LightGBM
- Random Forest
- Optional ensemble model

After disease prediction, advisory agents estimate disease severity percentage, explain symptoms, and provide suitable treatment or prevention suggestions based on the predicted disease and severity level.

---

# 2. Main Objectives

The project aims to:

1. Build an ML-only apple leaf disease classification system.
2. Classify apple leaf images into nine AppleLeaf9 categories.
3. Use handcrafted visual features instead of CNN or deep learning features.
4. Train and compare multiple classical ML models.
5. Build an ensemble classifier using voting or stacking.
6. Add pre-classification AI agents for image validation.
7. Add post-classification AI agents for severity, symptoms, and solution guidance.
8. Deploy the full system in a Streamlit web app.
9. Keep the system lightweight, explainable, and CPU-friendly.

---

# 3. AppleLeaf9 Categories

The project is based on nine AppleLeaf9 classes:

| Class Number | Category |
|---:|---|
| 1 | Healthy |
| 2 | Alternaria leaf spot |
| 3 | Brown spot |
| 4 | Frogeye leaf spot |
| 5 | Grey spot |
| 6 | Mosaic |
| 7 | Powdery mildew |
| 8 | Rust |
| 9 | Scab |

---

# 4. Complete System Workflow

```text
User uploads image
        ↓
Pre-classification validation agents
        ↓
Quality Agent
        ↓
Leaf Presence Agent
        ↓
Apple-Leaf Validator Agent
        ↓
Policy Agent
        ↓
If rejected: show reason and guidance
If accepted: continue
        ↓
Image preprocessing
        ↓
Feature extraction
        ↓
ML disease classifier
        ↓
Disease class + confidence
        ↓
Severity Estimation Agent
        ↓
Symptom Explanation Agent
        ↓
Treatment Recommendation Agent
        ↓
Final diagnosis report
```

---

# 5. Stage 1: Dataset Preparation

## 5.1 Goal

Prepare the AppleLeaf9 dataset in a clean, class-wise folder structure.

## 5.2 Expected Dataset Structure

```text
data/
  raw/
    AppleLeaf9/
      Healthy/
      Alternaria_leaf_spot/
      Brown_spot/
      Frogeye_leaf_spot/
      Grey_spot/
      Mosaic/
      Powdery_mildew/
      Rust/
      Scab/
```

## 5.3 Tasks

1. Download or collect the AppleLeaf9 dataset.
2. Organize images into nine class folders.
3. Remove unreadable or corrupted images.
4. Create a dataset manifest containing:
   - image path
   - disease label
   - image source
   - split name
   - validation tag if available
5. Create stratified train, validation, and test splits.

## 5.4 Output Files

```text
data/interim/manifest.csv
data/interim/splits/train.csv
data/interim/splits/val.csv
data/interim/splits/test.csv
reports/class_distribution.csv
```

## 5.5 Verification

Before moving forward:

- All nine classes must be present.
- Each image path in the manifest must exist.
- Train, validation, and test splits must not overlap.
- Every class should appear in every split.

---

# 6. Stage 2: Image Preprocessing

## 6.1 Goal

Standardize all images before feature extraction.

## 6.2 Preprocessing Steps

1. Read image using OpenCV.
2. Resize image to a fixed size, for example 256 × 256.
3. Convert image into required color spaces:
   - RGB
   - HSV
   - Grayscale
4. Optionally segment the leaf from the background.
5. Handle noisy images and edge cases safely.

## 6.3 Optional Leaf Segmentation

Leaf segmentation can be performed using:

- HSV thresholding
- green mask extraction
- morphological operations
- largest contour detection

## 6.4 Output

Preprocessing functions should be implemented in:

```text
src/preprocess.py
src/segment.py
```

## 6.5 Verification

- Every dataset image should preprocess without crashing.
- Segmentation masks should roughly cover the leaf area.
- Poor images should fail gracefully, not break the pipeline.

---

# 7. Stage 3: Handcrafted Feature Extraction

## 7.1 Goal

Convert each image into an explainable numerical feature vector.

No deep learning features are used.

## 7.2 Feature Families

### 7.2.1 Color Features

Color features capture discoloration caused by diseases.

Examples:

- RGB mean
- RGB standard deviation
- HSV mean
- HSV standard deviation
- Color histogram

Useful for:

- rust marks
- yellowing
- brown spots
- pale mildew regions
- healthy green leaf regions

### 7.2.2 Texture Features

Texture features capture surface irregularities.

Examples:

- GLCM contrast
- GLCM energy
- GLCM homogeneity
- GLCM correlation
- LBP histogram

Useful for:

- powdery mildew
- scab texture
- spot patterns
- lesion roughness

### 7.2.3 Shape Features

Shape features describe leaf or lesion geometry.

Examples:

- area
- perimeter
- circularity
- aspect ratio
- contour extent

Useful for:

- lesion shape
- spot size
- visible infected region

### 7.2.4 Gradient Features

Gradient features capture edge and structure information.

Example:

- HOG, Histogram of Oriented Gradients

Useful for:

- vein structure
- leaf boundary
- lesion edges
- disease pattern boundaries

## 7.3 Output

Feature extraction should be implemented in:

```text
src/features.py
src/feature_pipeline.py
```

Extracted features should be saved as:

```text
data/processed/features_train.npy
data/processed/features_val.npy
data/processed/features_test.npy
data/processed/y_train.npy
data/processed/y_val.npy
data/processed/y_test.npy
```

## 7.4 Verification

- Feature vector length must be fixed.
- No NaN values should exist.
- No infinite values should exist.
- Feature extraction time should be logged.
- The feature pipeline version should be recorded.

---

# 8. Stage 4: Feature Preprocessing

## 8.1 Goal

Prepare extracted features for ML models.

## 8.2 Steps

1. Encode disease labels using LabelEncoder.
2. Fit StandardScaler on training features only.
3. Transform train, validation, and test features using the same scaler.
4. Optionally apply PCA for dimensionality reduction.
5. Save scaler, label encoder, and PCA if used.

## 8.3 Important Rule

The scaler and PCA must be fitted only on the training split.

Do not fit scaler or PCA on validation or test data. That would cause data leakage.

## 8.4 Output Files

```text
models/scaler.pkl
models/label_encoder.pkl
models/pca.pkl
```

---

# 9. Stage 5: ML Model Training

## 9.1 Goal

Train four classical machine learning models for nine-class apple leaf disease classification.

## 9.2 Models

| Model | Purpose |
|---|---|
| KNN | Simple distance-based baseline |
| SVM | Strong margin-based classifier |
| LightGBM | Fast gradient boosting model |
| Random Forest | Robust tree ensemble model |

## 9.3 Training Steps

1. Load processed feature arrays.
2. Load encoded labels.
3. Train KNN.
4. Train SVM with RBF kernel and probability enabled.
5. Train LightGBM.
6. Train Random Forest.
7. Tune models using RandomizedSearchCV or GridSearchCV.
8. Evaluate on validation split.
9. Save best model from each algorithm.

## 9.4 Output Files

```text
models/knn.pkl
models/svm.pkl
models/lightgbm.pkl
models/random_forest.pkl
reports/model_validation_results.csv
```

## 9.5 Verification

- All four models should train successfully.
- All saved model files should reload correctly.
- Each model should beat a majority-class baseline.
- Validation macro-F1 should be recorded.

---

# 10. Stage 6: Ensemble Model

## 10.1 Goal

Combine multiple ML models to improve robustness.

## 10.2 Ensemble Options

### Option 1: Soft Voting

Soft voting averages predicted probabilities from multiple models.

```text
Final probability = average of KNN, SVM, LightGBM, and Random Forest probabilities
```

### Option 2: Stacking

Stacking trains a meta-model using the predictions of base models.

Example meta-model:

```text
Logistic Regression
```

## 10.3 Output Files

```text
models/voting_ensemble.pkl
models/stacking_ensemble.pkl
reports/ensemble_results.csv
reports/best_model_selection.json
```

## 10.4 Verification

- Voting model should be evaluated.
- Stacking model should be evaluated.
- Best final model should be selected based on validation macro-F1.
- If ensemble does not improve performance, the best individual model can be selected honestly.

---

# 11. Stage 7: ML Core Evaluation

## 11.1 Goal

Evaluate the disease classifier before adding the agentic validation layer.

This gives the baseline performance of the ML core.

## 11.2 Evaluation Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Overall correctness |
| Macro-F1 | Balanced class performance |
| Precision | Correctness of positive predictions |
| Recall | Ability to find actual class samples |
| Confusion Matrix | Shows class-wise confusion |
| Inference Time | Measures deployment practicality |

## 11.3 Output Files

```text
reports/disease_metrics_ungated.csv
reports/per_class_report_ungated.csv
reports/confusion_matrix_ungated.png
reports/inference_time_ungated.csv
```

## 11.4 Verification

- Test metrics should be saved.
- Confusion matrix should be reviewed.
- Weak disease classes should be identified.
- This result becomes the baseline for comparison with the agentic system.

---

# 12. Stage 8: Pre-Classification AI Agents

## 12.1 Goal

Reject bad or invalid images before disease prediction.

This prevents the classifier from giving forced predictions on poor-quality, non-leaf, or wrong-species images.

---

## 12.2 Agent 1: Quality Agent

### Purpose

Checks whether the image quality is good enough for disease diagnosis.

### Checks

- blur
- brightness
- contrast
- resolution

### Possible Methods

- Laplacian variance for blur detection
- mean brightness check
- contrast threshold
- minimum width and height check

### Output

```json
{
  "quality_pass": true,
  "quality_score": 0.91,
  "quality_reason": "Image is clear and bright enough for diagnosis."
}
```

### Failure Example

```json
{
  "quality_pass": false,
  "quality_score": 0.32,
  "quality_reason": "Image is too blurry. Please upload a clearer image."
}
```

---

## 12.3 Agent 2: Leaf Presence Agent

### Purpose

Checks whether the uploaded image contains a leaf-like object.

### Checks

- green or leaf-colored region
- contour area
- leaf-like shape
- mask quality

### Possible Methods

- HSV green mask
- contour detection
- largest contour area
- shape sanity checks

### Output

```json
{
  "leaf_present": true,
  "leaf_bbox": [45, 30, 210, 230],
  "mask_quality": 0.84,
  "reason": "A leaf-like region was detected."
}
```

### Failure Example

```json
{
  "leaf_present": false,
  "leaf_bbox": null,
  "mask_quality": 0.12,
  "reason": "No clear leaf region was detected."
}
```

---

## 12.4 Agent 3: Apple-Leaf Validator Agent

### Purpose

Checks whether the detected leaf is likely an apple leaf.

### Method

This agent can use a small binary ML classifier:

```text
apple leaf vs. non-apple leaf
```

It can reuse the same handcrafted feature pipeline used by the disease classifier.

### Training Data

This agent requires:

- positive samples: apple leaf images
- negative samples: non-apple leaves or non-leaf images

### Output

```json
{
  "is_apple_leaf": true,
  "confidence": 0.88,
  "reason": "The image is likely an apple leaf."
}
```

### Failure Example

```json
{
  "is_apple_leaf": false,
  "confidence": 0.41,
  "reason": "The detected leaf does not appear to be an apple leaf."
}
```

---

## 12.5 Agent 4: Policy Agent

### Purpose

Combines decisions from the previous agents and decides whether the image should continue to disease classification.

### Decision Rules

```text
If Quality Agent fails:
    reject image
Else if Leaf Presence Agent fails:
    reject image
Else if Apple-Leaf Validator confidence is below threshold:
    reject image or flag for review
Else:
    accept image and send to disease classifier
```

### Output if Accepted

```json
{
  "accepted": true,
  "decision": "pass_to_classifier",
  "reason": "Image passed all validation checks."
}
```

### Output if Rejected

```json
{
  "accepted": false,
  "decision": "reject",
  "reason": "Image is too blurry for reliable diagnosis."
}
```

## 12.6 Agent Files

```text
src/agents/quality_agent.py
src/agents/leaf_presence_agent.py
src/agents/apple_leaf_agent.py
src/agents/policy_agent.py
src/agents/orchestrator.py
```

## 12.7 Verification

- Each agent must be testable independently.
- The orchestrator should stop at the first failure.
- Every rejection must include a clear reason.
- Good apple leaf images should pass validation.
- Bad, blurry, non-leaf, and wrong-species images should be rejected.

---

# 13. Stage 9: Disease Classification After Validation

## 13.1 Goal

Classify disease only after image validation passes.

## 13.2 Steps

1. Receive accepted image from Policy Agent.
2. Preprocess image.
3. Extract handcrafted features.
4. Apply scaler and PCA if used.
5. Run best disease classifier or ensemble.
6. Return disease class and confidence.

## 13.3 Output Example

```json
{
  "disease_class": "Rust",
  "confidence": 0.92,
  "model_used": "Voting Ensemble"
}
```

---

# 14. Stage 10: Post-Classification Advisory Agents

After disease prediction, additional agents provide practical explanation and guidance.

---

## 14.1 Agent 5: Severity Estimation Agent

### Purpose

Estimate the visible percentage of disease on the leaf.

### Method

The agent can estimate infected area using image processing:

```text
Disease percentage = infected leaf area / total leaf area × 100
```

### Steps

1. Segment total leaf area.
2. Detect diseased regions using color and texture heuristics.
3. Count infected pixels.
4. Count total leaf pixels.
5. Calculate disease percentage.
6. Map percentage to severity level.

### Severity Levels

| Disease Percentage | Severity Level |
|---:|---|
| 0% | Healthy |
| 1–10% | Mild |
| 11–30% | Moderate |
| 31–60% | Severe |
| Above 60% | Critical |

### Output Example

```json
{
  "disease_percentage": 24.7,
  "severity_level": "Moderate",
  "reason": "Approximately one quarter of the visible leaf area appears affected."
}
```

---

## 14.2 Agent 6: Symptom Explanation Agent

### Purpose

Explain the predicted disease in simple language.

### Input

- disease class
- confidence
- severity level
- disease percentage

### Output Example

```json
{
  "symptoms": [
    "Orange or rusty spots on the leaf surface",
    "Yellowing around infected regions",
    "Reduced leaf health if infection spreads"
  ]
}
```

---

## 14.3 Agent 7: Treatment Recommendation Agent

### Purpose

Provide suitable prevention and treatment suggestions.

### Input

- disease class
- severity level
- disease percentage

### Output Example

```json
{
  "recommendations": [
    "Remove heavily infected leaves if possible.",
    "Improve airflow around the tree.",
    "Avoid overhead watering.",
    "Consult a local agricultural expert for fungicide selection if infection is severe."
  ]
}
```

## 14.4 Agent 8: Report Agent

### Purpose

Combine all outputs into a clear final report.

### Final Report Includes

- validation result
- disease class
- confidence score
- model used
- estimated disease percentage
- severity level
- symptoms
- recommendations

### Output Example

```json
{
  "status": "diagnosis_complete",
  "disease": "Rust",
  "confidence": 0.92,
  "severity": "Moderate",
  "disease_percentage": 24.7,
  "symptoms": ["Rust-colored spots", "Yellowing around affected area"],
  "solutions": ["Improve airflow", "Remove infected leaves", "Consult expert for treatment"]
}
```

---

# 15. Stage 11: Agentic Pipeline Evaluation

## 15.1 Goal

Measure how well the validation and advisory agents work.

## 15.2 Validation Agent Metrics

| Agent | Metric |
|---|---|
| Quality Agent | precision, recall |
| Leaf Presence Agent | accuracy |
| Apple-Leaf Validator Agent | binary F1-score |
| Policy Agent | false rejection rate |

## 15.3 Disease Classifier With Gate

Evaluate the disease classifier only on images accepted by the validation agents.

Metrics:

- gated accuracy
- gated macro-F1
- gated precision
- gated recall
- gated confusion matrix

## 15.4 Most Important Comparison

Compare:

```text
Ungated disease classifier performance
vs.
Gated disease classifier performance
```

This proves whether the agentic layer improves real-world reliability.

## 15.5 Output Files

```text
reports/validation_agent_metrics.csv
reports/disease_metrics_gated.csv
reports/confusion_matrix_gated.png
reports/gated_vs_ungated_comparison.csv
```

---

# 16. Stage 12: Streamlit Web Application

## 16.1 Goal

Build a simple web app where users can upload apple leaf images and receive either a rejection reason or a disease diagnosis.

## 16.2 User Flow

1. User uploads image.
2. App displays uploaded image.
3. Validation agents run.
4. App shows validation checklist.
5. If validation fails, app shows reason and advice.
6. If validation passes, disease classifier runs.
7. App shows disease prediction, confidence, severity percentage, symptoms, and solutions.

## 16.3 Suggested App Tabs

| Tab | Purpose |
|---|---|
| Upload Image | User uploads leaf image |
| Validation Trace | Shows agent pass/fail results |
| Disease Prediction | Shows disease class and confidence |
| Severity and Solution | Shows disease percentage, symptoms, and recommendations |
| Model Info | Shows model used and feature pipeline |

## 16.4 Main App File

```text
app.py
```

## 16.5 Run Command

```bash
streamlit run app.py
```

---

# 17. Suggested Folder Structure

```text
AgriGuard-ML/
├── app.py
├── requirements.txt
├── README.md
├── configs/
│   ├── paths.yaml
│   ├── model_config.yaml
│   └── agent_thresholds.yaml
├── data/
│   ├── raw/
│   │   └── AppleLeaf9/
│   ├── interim/
│   │   ├── manifest.csv
│   │   └── splits/
│   └── processed/
├── docs/
│   ├── scope.md
│   ├── acceptance_criteria.md
│   └── dataset_notes.md
├── models/
│   ├── knn.pkl
│   ├── svm.pkl
│   ├── lightgbm.pkl
│   ├── random_forest.pkl
│   ├── voting_ensemble.pkl
│   ├── stacking_ensemble.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── pca.pkl
├── reports/
│   ├── disease_metrics_ungated.csv
│   ├── disease_metrics_gated.csv
│   ├── validation_agent_metrics.csv
│   ├── confusion_matrix_ungated.png
│   ├── confusion_matrix_gated.png
│   └── gated_vs_ungated_comparison.csv
├── src/
│   ├── preprocess.py
│   ├── segment.py
│   ├── features.py
│   ├── feature_pipeline.py
│   ├── train.py
│   ├── ensemble.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── severity.py
│   ├── advisory.py
│   └── agents/
│       ├── quality_agent.py
│       ├── leaf_presence_agent.py
│       ├── apple_leaf_agent.py
│       ├── policy_agent.py
│       ├── severity_agent.py
│       ├── symptom_agent.py
│       ├── treatment_agent.py
│       ├── report_agent.py
│       └── orchestrator.py
└── tests/
    ├── test_features.py
    ├── test_preprocess.py
    ├── test_agents.py
    ├── test_predict.py
    └── test_pipeline.py
```

---

# 18. Suggested Requirements

```text
numpy>=1.20.0
pandas>=1.3.0
opencv-python>=4.5.0
scikit-image>=0.18.0
scikit-learn>=1.0.0
lightgbm>=4.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.1.0
tqdm>=4.60.0
streamlit>=1.20.0
Pillow>=8.0.0
PyYAML>=6.0
pytest>=7.0.0
```

---

# 19. Execution Order

## Step 1: Prepare Dataset

```bash
python src/make_manifest.py
python src/make_splits.py
```

## Step 2: Extract Features

```bash
python src/feature_pipeline.py
```

## Step 3: Train Individual Models

```bash
python src/train.py
```

## Step 4: Train Ensemble Models

```bash
python src/ensemble.py
```

## Step 5: Evaluate ML Core

```bash
python src/evaluate.py --mode ungated
```

## Step 6: Evaluate Agents

```bash
python src/evaluate_agents.py
```

## Step 7: Evaluate Full Gated Pipeline

```bash
python src/evaluate.py --mode gated
```

## Step 8: Run Streamlit App

```bash
streamlit run app.py
```

---

# 20. Final Output Shown to User

If image fails validation:

```text
Status: Rejected
Reason: Image is too blurry for reliable disease diagnosis.
Suggestion: Please upload a clearer image in better lighting.
```

If image passes validation:

```text
Status: Accepted
Disease Detected: Rust
Confidence: 92%
Estimated Disease Area: 24.7%
Severity Level: Moderate
Symptoms: Rust-colored spots, yellowing around affected regions
Suggested Action: Remove infected leaves, improve airflow, avoid overhead watering, and consult a local agricultural expert if infection spreads.
Model Used: Voting Ensemble
```

---

# 21. Testing Plan

## 21.1 Unit Tests

Test:

- preprocessing output shape
- feature vector length
- no NaN or infinite values
- quality agent decision
- leaf presence agent decision
- apple-leaf validator output
- policy agent logic
- severity percentage calculation

## 21.2 Integration Tests

Test:

- valid apple leaf image goes through full pipeline
- blurry image is rejected
- non-leaf image is rejected
- non-apple leaf image is rejected
- accepted image returns disease result

## 21.3 Performance Tests

Measure:

- average preprocessing time
- average feature extraction time
- average model prediction time
- full pipeline inference time

---

# 22. Evaluation Summary Table

| Component | Evaluation Metric |
|---|---|
| KNN | accuracy, macro-F1, inference time |
| SVM | accuracy, macro-F1, inference time |
| LightGBM | accuracy, macro-F1, inference time |
| Random Forest | accuracy, macro-F1, inference time |
| Ensemble | accuracy, macro-F1, inference time |
| Quality Agent | precision, recall |
| Leaf Presence Agent | accuracy |
| Apple-Leaf Validator Agent | binary F1-score |
| Policy Agent | false rejection rate |
| Severity Agent | severity estimation error if ground truth exists |
| Full Pipeline | gated accuracy, rejection rate, user-facing correctness |

---

# 23. Key Rules for the Project

1. Do not use deep learning in the main system.
2. Use handcrafted features only.
3. Fit scaler and PCA only on training data.
4. Validate image before disease classification.
5. Always provide a human-readable rejection reason.
6. Always provide confidence with disease prediction.
7. Disease percentage should be treated as an estimate unless pixel-level ground truth is available.
8. Treatment suggestions should be general guidance, not a replacement for expert agricultural advice.
9. Keep validation, classification, severity estimation, and advisory logic modular.
10. Compare ungated and gated performance to prove the value of agents.

---

# 24. Future Improvements

Possible future additions:

- Better disease severity estimation using segmentation masks.
- More accurate apple-leaf validator using more non-apple leaf examples.
- Multi-species expansion beyond apple leaves.
- Active learning for uncertain predictions.
- Offline mobile deployment.
- Farmer-friendly multilingual interface.
- Expert-verified treatment recommendation database.

---

# 25. Final Definition of Success

The project is successful when:

1. The system classifies apple leaves into nine AppleLeaf9 categories.
2. At least four classical ML models are trained and compared.
3. The best model or ensemble is selected using validation and test metrics.
4. Invalid images are rejected before classification.
5. Valid images receive disease class, confidence, severity percentage, symptoms, and treatment guidance.
6. The Streamlit app runs end-to-end.
7. Reports clearly compare ML-only performance with and without the agentic validation layer.

---

# 26. Project Summary

AgriGuard-ML is a complete ML-only and agentic apple leaf disease detection framework. It uses handcrafted features and classical ML models to detect nine AppleLeaf9 disease categories. It improves reliability by adding validation agents before classification and advisory agents after classification. The final system is lightweight, explainable, modular, and suitable for real-world agricultural use where uploaded images may be blurry, invalid, or taken in uncontrolled environments.
