# Project Synopsis

## Project Title
**AgriGuard-ML: An Agentic Machine Learning Framework for Apple Leaf Disease Detection with Pre-Classification Leaf Validation**

---

## 1. Introduction

Apple cultivation is affected by several leaf diseases — such as scab and rust — that reduce yield and fruit quality if not identified and treated early. Traditionally, disease identification is done manually by farmers or agronomists through visual inspection, a process that is slow, subjective, and dependent on expert availability. As smartphone cameras and low-cost computing have become widely accessible, automated image-based disease detection has emerged as a practical alternative, particularly for farmers in regions with limited access to agricultural experts.

Most existing automated approaches rely on deep learning models such as convolutional neural networks (CNNs). While these models can achieve high accuracy, they come with real drawbacks for field deployment: they require large labeled datasets, substantial computing power (often GPUs) for both training and inference, and their decisions are largely opaque — it is difficult to explain *why* a CNN predicted a particular disease class. This lack of interpretability is a real barrier for agricultural applications, where a farmer or extension worker may want to understand the visual basis of a prediction before acting on it.

This project proposes an **ML-only** approach: instead of learning features automatically through a deep network, it extracts well-understood, handcrafted visual descriptors — color, texture, shape, and gradient features — from leaf images, and classifies them using classical machine learning models (K-Nearest Neighbors, Support Vector Machine, LightGBM, and Random Forest), optionally combined through a voting or stacking ensemble. This approach is lighter-weight, faster to train and deploy, and more interpretable, since each feature has a direct, explainable relationship to visual leaf characteristics like color variation, surface texture, and lesion shape.

A second, equally important problem is addressed alongside disease classification: most systems assume the uploaded image is already a valid, in-focus apple leaf, and predict a disease regardless of whether that assumption holds. In practice, users upload blurry photos, images with no leaf at all, or leaves from an entirely different plant species — and a disease classifier will still confidently return a prediction for any of these. To address this, the project introduces an **agentic validation layer**: a small pipeline of independent decision-making agents that check image quality, confirm the presence of a leaf, and verify that the leaf is specifically an apple leaf, *before* the image is ever passed to the disease classifier. Only images that pass this gate are diagnosed; everything else is rejected with a clear, human-readable reason. This combination — an interpretable ML disease classifier guarded by an agentic input-validation pipeline — is what distinguishes this project from a standard classify-anything image pipeline.

---

## 2. Literature Review

Automated plant disease detection has been studied extensively over the past decade, broadly falling into two categories: classical machine learning approaches using handcrafted features, and deep learning approaches using learned representations.

**Handcrafted feature-based approaches.** Early work in plant disease classification relied on color, texture, and shape descriptors combined with classical classifiers such as Support Vector Machines and k-Nearest Neighbors. Color-based features (RGB/HSV statistics, color histograms) are effective because diseased regions often show distinct discoloration compared to healthy tissue. Texture descriptors such as the Gray-Level Co-occurrence Matrix (GLCM) and Local Binary Patterns (LBP) capture surface irregularities associated with lesions, spots, and fungal growth, which are often not well captured by color alone. Shape-based features (area, perimeter, circularity) are useful once a lesion or the leaf itself has been segmented from the background. These approaches are computationally inexpensive and remain competitive on small-to-medium sized datasets, though their accuracy typically depends heavily on the quality of feature engineering.

**Deep learning approaches.** More recent work has largely shifted toward CNN-based architectures (e.g., ResNet, EfficientNet, MobileNet) applied directly to raw leaf images, often achieving higher raw accuracy on large public datasets such as PlantVillage. However, this line of work generally requires larger datasets, longer training times, and more computational resources, and the resulting models function largely as black boxes — a known concern in applied agricultural settings where trust and explainability matter.

**Ensemble methods.** Across both paradigms, ensemble techniques — combining multiple base models through voting or stacking — have consistently been shown to improve robustness and reduce variance compared to any single model, at a relatively low additional computational cost. This motivates the choice, in this project, of an ensemble over classical models rather than relying on a single classifier.

**Input validation and agentic pipelines.** A separate but related thread of work looks at pipeline reliability rather than raw classification accuracy — the observation that most published disease-detection systems are evaluated on curated benchmark datasets, and implicitly assume every input image is a valid, in-focus, correctly-framed leaf of the target species. Real-world deployments do not offer this guarantee. Agentic AI — the use of small, independently reasoning modules ("agents") that each perform a narrow check and hand off a decision to a coordinating policy — has increasingly been used in applied ML systems to add input-validation and guardrail stages ahead of a core model, rather than folding every concern into one monolithic classifier. Applied to this domain, such an approach allows image-quality checks, leaf-presence checks, and species checks to be built, tested, and reasoned about independently of the disease classifier itself, and lets the system reject or flag unreliable inputs with an explicit reason rather than silently producing a low-confidence guess.

**Gap identified.** Much of the classical-ML literature evaluates only one or two feature families or one or two classifiers in isolation, rather than combining a broad, complementary feature set (color + texture + shape + gradient) with a multi-model ensemble in a single, end-to-end pipeline specific to apple leaf diseases. Separately, most disease-detection systems — classical or deep learning — do not validate the input before diagnosing it, leaving them vulnerable to non-leaf, wrong-species, or poor-quality images. This project addresses both gaps together: it systematically combines four complementary handcrafted feature families with four classical models and an ensemble layer, and wraps that classifier in an agentic validation pipeline that checks image quality, leaf presence, and apple-leaf identity before any diagnosis is made — a combination not commonly found as a single, evaluated pipeline in the existing literature.

---

## 3. Objectives

The primary objective of this project is to design, build, and evaluate an ML-only pipeline for apple leaf disease detection. Specifically, the project aims to:

1. Build a labeled, class-organized dataset pipeline with stratified train/validation/test splits for apple leaf images.
2. Extract a comprehensive handcrafted feature set — color, texture (GLCM, LBP), shape, and gradient (HOG) features — from each leaf image, without using deep learning.
3. Train and tune four classical machine learning models — KNN, SVM, LightGBM, and Random Forest — as independent disease classifiers.
4. Combine the four base models into a single ensemble classifier using soft voting and/or stacking, and compare its performance against the individual models.
5. Evaluate all models using standard classification metrics (accuracy, macro-F1, per-class precision/recall, confusion matrix) as well as practical metrics (inference time).
6. Design and implement an **agentic validation pipeline** consisting of a Quality Agent, a Leaf Presence Agent, an Apple-Leaf Validator Agent, and a Policy Agent, that inspects an uploaded image and decides — with an explicit, human-readable reason — whether it is eligible for disease diagnosis.
7. Quantify the impact of the agentic validation layer by comparing disease-classification performance with and without input validation, and by measuring the validation pipeline's own accuracy (quality-check precision/recall, leaf-presence accuracy, apple-leaf binary F1) and its false-rejection rate.
8. Deploy the complete system — agentic validation followed by the best-performing disease model (single or ensemble) — in a Streamlit web application that allows a user to upload a leaf image and receive either a validation-failure reason or a predicted disease class with a confidence score.

---

## 4. Methodology

### 4.1 Dataset Collection
Labeled apple leaf images are organized by class folders (e.g., Healthy, Scab, Rust, Multiple Diseases), sourced from a public plant-disease image repository. Each image is associated with a single disease label. The dataset is split into training, validation, and test sets using **stratified sampling**, so that the class distribution is preserved across all three splits and no split is dominated or starved of any particular class.

### 4.2 Data Pre-processing and Augmentation
Before feature extraction, each image undergoes a standard pre-processing pipeline:

- **Resizing** — all images are resized to a fixed resolution (e.g., 256×256) to ensure consistent feature dimensionality across the dataset.
- **Color-space conversion** — images are converted between RGB, HSV, and grayscale as required by different feature extractors (e.g., HSV for color histograms, grayscale for GLCM/LBP/HOG).
- **Optional segmentation** — the leaf is optionally isolated from its background using HSV thresholding and morphological operations, followed by extraction of the largest contour. This improves the reliability of shape-based features and reduces background noise in color/texture features.
- **Augmentation** — where the dataset is limited in size for certain classes, standard augmentation techniques (rotation, flipping, brightness/contrast jitter) may be applied to the training split only, to improve model generalization without introducing data leakage into validation/test sets.

### 4.3 Overview of Model (Feature Extraction + Classification)

**Feature extraction (handcrafted, no deep learning):**
- *Color features* — RGB/HSV channel mean and standard deviation, and color histograms, capturing discoloration patterns associated with disease.
- *Texture features* — GLCM statistics (contrast, energy, homogeneity, correlation) and LBP histograms, capturing surface irregularity and lesion texture.
- *Shape features* — area, perimeter, and circularity of the segmented leaf/lesion region, where segmentation is available.
- *Gradient features* — HOG (Histogram of Oriented Gradients) descriptors, capturing edge and gradient structure across the leaf surface.

All feature families are concatenated into a single feature vector per image.

**Feature preprocessing:**
- Features are standardized using `StandardScaler` (fit on the training split only), which is particularly important for distance- and margin-based models like KNN and SVM.
- Optional dimensionality reduction via PCA, also fit on the training split only.
- Class labels are encoded numerically using `LabelEncoder`.

**Classification models:**
Four classical ML models are trained independently on the processed feature vectors:
1. **K-Nearest Neighbors (KNN)**
2. **Support Vector Machine (SVM)** — RBF kernel, with probability estimates enabled
3. **LightGBM** — gradient-boosted decision trees
4. **Random Forest**

Each model is tuned using cross-validation and randomized hyperparameter search (`RandomizedSearchCV`) on the training split.

**Ensemble layer:**
The four base models are combined into a single ML-only hybrid predictor using either:
- **Soft voting** — averaging predicted class probabilities across all four models, or
- **Stacking** — training a Logistic Regression meta-model on the base models' outputs.

The ensemble is compared against the best individual model to determine the final deployed model.

### 4.4 Agentic Validation Layer
Once the disease classifier is trained and proven on its own, it is wrapped in an agentic validation pipeline that runs **before** any diagnosis, so that only trustworthy inputs reach the classifier. The pipeline consists of four cooperating agents:

- **Quality Agent** — checks blur (via Laplacian variance), illumination, and minimum resolution, and outputs a pass/fail decision with a reason.
- **Leaf Presence Agent** — uses a green-mask and contour-based heuristic (plus basic texture/shape sanity checks) to determine whether a leaf-like region is present in the image at all.
- **Apple-Leaf Validator Agent** — a small binary ML classifier, built on the same handcrafted feature pipeline used for disease classification, that distinguishes apple leaves from leaves of other species, and reports a confidence score.
- **Policy Agent** — combines the outputs of the previous three agents using explicit rules (quality fail → reject; leaf-absence fail → reject; low apple-leaf confidence → reject or flag for review; otherwise → forward to the disease classifier) and produces the final accept/reject decision along with a human-readable reason.

This keeps input validation and disease diagnosis as two separable, independently testable stages, rather than one entangled model.

### 4.5 Evaluation
Evaluation is carried out at two levels, so that the contribution of the agentic layer can be measured separately from the disease classifier itself:

**Disease classifier (Level 1).** All four individual classifiers and the ensemble(s) are evaluated on a held-out test set using:
- **Accuracy**
- **Macro-F1 score** (to fairly weight performance across all disease classes, including minority classes)
- **Per-class precision and recall**
- **Confusion matrix** (to identify which disease classes are most often confused with one another)
- **Inference time** (to confirm practicality for real-time or near-real-time use in the Streamlit app)

**Agentic validation layer (Level 2).** The validation agents are evaluated on a separate held-out set that includes non-leaf and non-apple-leaf images in addition to genuine apple leaf images, using:
- **Quality-check precision/recall**
- **Leaf-presence accuracy**
- **Apple-leaf binary classification F1**
- **False-rejection rate** — the rate at which genuinely valid apple leaf images are incorrectly rejected, which is the most important metric for user trust

**End-to-end comparison.** Disease-classification accuracy is measured both *without* the validation gate (Level 1 alone) and *with* it (only on images the agentic layer accepts), which provides direct, quantified evidence of whether the agentic validation layer improves real-world reliability.

The best-performing configuration — whether a single model or an ensemble, evaluated with the agentic layer in front of it — is selected for deployment based on this combined evaluation.

---

## 5. Expected Outcomes

- A working, reproducible ML-only pipeline capable of classifying apple leaf images into disease categories using only handcrafted features and classical models — with no deep learning dependency.
- Quantified comparison of KNN, SVM, LightGBM, and Random Forest on the same feature set, identifying which classical model performs best for this task.
- Evidence of whether a voting or stacking ensemble meaningfully improves accuracy and robustness over the single best individual model.
- A lightweight, interpretable, and relatively fast disease-classification system — with inference times low enough for practical, real-time use in a lightweight Streamlit application — suitable for deployment in resource-constrained environments (e.g., low-cost devices or offline field use) where GPU-based deep learning is impractical.
- A working agentic validation pipeline (Quality, Leaf Presence, Apple-Leaf Validator, and Policy Agents) that reliably rejects blurry, non-leaf, or wrong-species images with a clear, explainable reason, rather than forcing a disease prediction on unreliable input.
- Quantified evidence of the agentic layer's real-world value: measured false-rejection rate, and a direct before/after comparison showing how disease-classification accuracy on accepted images compares to accuracy without any input validation.
- A deployed Streamlit application where a user can upload an apple leaf image and either (a) receive actionable feedback if the image fails validation, or (b) receive a predicted disease class along with a confidence score and the identity of the model used, once the image passes validation.
- A documented feature-importance and confusion-matrix analysis giving insight into which visual characteristics (color, texture, shape) are most useful for distinguishing between disease classes — insight that a black-box deep learning model would not directly provide.

---

## 6. Conclusion

This project demonstrates that apple leaf disease detection does not necessarily require deep learning to be effective, and that reliability against real-world, messy input does not require abandoning that simplicity either. By combining a broad set of handcrafted visual features — color, texture, shape, and gradient descriptors — with well-tuned classical machine learning models and an ensemble layer, it is possible to build a disease classifier that is lightweight, computationally inexpensive, and inherently more interpretable than a deep learning-based alternative. Wrapping that classifier in an agentic validation pipeline — which checks image quality, leaf presence, and apple-leaf identity before any diagnosis is attempted — closes a gap that most disease-detection systems leave open: the assumption that every uploaded image is a valid, on-topic leaf photo. Together, the two layers make the system well-suited to real-world agricultural deployment scenarios where computational resources are limited, input quality cannot be guaranteed, and interpretability of both predictions and rejections is valuable to end users such as farmers and agricultural extension workers. The resulting pipeline — from dataset preparation, feature extraction, model training, and ensembling, through agentic input validation, two-level evaluation, and deployment in a Streamlit application — provides a complete, reproducible, and explainable end-to-end system for trustworthy apple leaf disease detection.
