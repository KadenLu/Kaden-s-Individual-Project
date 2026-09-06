# MyAnimeList User Behavioral Archetypes & Predictive Modeling

This project explores behavioral patterns in anime consumption by clustering users based on their rating habits and predicting these archetypes using profile metadata and engagement statistics. The primary goal is to determine if engagement behavior alone can effectively identify user personas without relying on demographic features like age or gender.

---

## 🚀 Project Overview

Traditional user profiling often relies heavily on demographics. This study shifts the focus to **behavioral features**: how users interact with content and how their personal rating distributions differ from the global community.

### Key Research Question
> Can we predict user behavioral archetypes (personas) using only interaction metadata, and are demographic features like age and gender redundant for this task?

---

## 🛠️ Methodology & Pipeline

```text
Raw Data ➔ Feature Engineering (Vu, Du) ➔ In-Fold Clustering ➔ Classification Baseline ➔ Feature Ablation & TOST
```

### 1. Feature Engineering
We define two core behavioral features for every user:
* **User Rating Variance (Vu):** Measures the internal consistency and spread of a user's ratings. High variance indicates a user who utilizes the full range of scores (1–10), whereas low variance indicates a "flat" rater.
* **Average Relative Difference (Du):** Measures how a user's score for a given anime compares to the global community mean for that title. Positive values denote a lenient/generous rater, while negative values reflect a more critical rater.

### 2. Archetype Generation
A dual-clustering strategy establishes stable personas without heuristic labeling:
* **K-Means (k=3):** Discovered the primary archetypes—*Low score/high variance* (critical), *Average score/variance*, and *High score/low variance* (enthusiast). Selected based on Adjusted Rand Index (ARI) stability.
* **Gaussian Mixture Models (GMM, k=4):** Evaluated as a parametric robustness check to assess non-spherical clusters.

### 3. Leakage-Free Predictive Modeling
To prevent data leakage, clustering models were fitted strictly within training splits across a **Stratified 10-Fold Cross-Validation** pipeline:
* **Evaluated Classifiers:** Random Forest, LightGBM, XGBoost, Decision Tree
* **Baselines:** Majority Class and Stratified Prior Classifiers

### 4. Statistical Verification
Evaluated via **Balanced Accuracy (BA)** and **Macro F1 Score**:
* **Corrected Resampled t-Test:** Corrects for fold interdependence in repeated cross-validation splits.
* **Two One-Sided Tests (TOST):** Equivalence bounds set to ±0.02 (alpha = 0.05) to test practical predictive equivalence under feature ablation.

---

## 📊 Key Findings

| Metric / Finding | Outcome |
| :--- | :--- |
| **Top Performing Models** | **Random Forest** and **LightGBM** achieved the highest accuracy (~51–52% Balanced Accuracy on the 3-class problem), significantly outperforming baseline models (p < 0.0001). |
| **Demographic Redundancy** | Removing `age` and `gender` caused zero operational drop in performance. TOST confirmed predictive equivalence between behavioral-only models and demographic-augmented models. |
| **Primary Predictor** | Permutation importance identified **`user_dropped`** (drop rate) as the single most influential predictor of a user's rating archetype. |

---

## 📁 Repository Structure

```text
├── anime_behavioral_clustering.ipynb   # End-to-end analysis: data cleaning, clustering, & modeling
├── figure_1_histograms.png             # Distribution plots of Vu and Du metrics
├── workflow_pipeline.png               # Visual diagram of the leakage-free pipeline architecture
└── README.md                           # Project documentation
```

---

## 📦 Environment & Dependencies

* **Python:** `3.13+`
* **Core Libraries:** `scikit-learn`, `lightgbm`, `xgboost`, `pandas`, `numpy`, `scipy`
