# Human Activity Recognition (AReM)

> **Can we classify what a person is doing from wireless sensor signals alone?**  
> Time-series feature engineering pipeline on the UCI AReM dataset — extracting statistical descriptors from RSS measurements, then comparing Logistic Regression, L1-penalized LR, and Naïve Bayes classifiers across binary and multi-class settings.

---

## Results at a glance

### Binary classification (bending vs. other)

| Model | Accuracy | Notes |
|-------|----------|-------|
| Logistic Regression + RFE | ~0.97 | Best `(l, p)` via 5-fold StratifiedKFold |
| L1-Penalized LR (LASSO) | ~0.97 | λ auto-tuned via `LogisticRegressionCV` |

### Multi-class classification (7 activities)

| Model | Test Accuracy | Notes |
|-------|--------------|-------|
| **L1 Multinomial LR** | **best** | Handles correlated features via simultaneous selection + regularization |
| Gaussian Naïve Bayes | good | Assumes feature independence |
| Multinomial Naïve Bayes | weakest | Designed for count data — poor fit for continuous RSS features |

L1-penalized multinomial logistic regression is the recommended model: it performs variable selection and regularization in a single step, is numerically stable under near-perfect class separation, and generalises better than Naïve Bayes on correlated sensor features.

---

## Dataset

**AReM (Activity Recognition system based on Multisensor data fusion)**  
Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Activity+Recognition+system+based+on+Multisensor+data+fusion+%28AReM%29)

- 7 activities: `bending1`, `bending2`, `cycling`, `lying`, `sitting`, `standing`, `walking`
- 6 time series per instance: `avg_rss12`, `var_rss12`, `avg_rss13`, `var_rss13`, `avg_rss23`, `var_rss23`
- Train/test split follows the dataset convention:
  - `bending1`, `bending2`: files 1–2 → test, rest → train
  - All other activities: files 1–3 → test, rest → train

---

## Project structure

```
human-activity-recognition/
├── data/                        # AReM dataset auto-downloaded by notebook
├── 01_feature_engineering.ipynb # HW3: feature extraction + bootstrap CI
├── 02_classification.ipynb      # HW4: binary + multi-class models
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
pip install -r requirements.txt
jupyter notebook 01_feature_engineering.ipynb
# then open 02_classification.ipynb
```

The dataset is downloaded automatically from the UCI repository on first run — no manual setup needed.

---

## Pipeline overview

### Notebook 1 — Feature engineering (`01_feature_engineering.ipynb`)

**Time-domain feature extraction**  
For each instance, 7 statistical features are extracted per time series × 6 series = 42 features (at `l=1`):

| Feature | Description |
|---------|-------------|
| Min | Smallest value |
| Max | Largest value |
| Mean | Average |
| Median | 50th percentile |
| Std dev | Spread |
| 1st quartile | 25th percentile |
| 3rd quartile | 75th percentile |

**Segment parameter `l`**  
Each time series can be split into `l` equal segments before feature extraction. `l=2` doubles the feature count to 84. Cross-validation over `l ∈ {1, 2, 3, 5}` shows `l=1` is sufficient — global statistics already capture the discriminative signal.

**Bootstrap confidence intervals (90%)**  
Standard deviation estimates for each feature are bootstrapped (1,000 resamples) to produce 90% CIs, validating that the extracted statistics are stable across different training draws.

**Feature importance**  
Top 3 features selected by between-class variance (proxy for discriminative power):  
`ts1_min`, `ts1_max`, and `ts1_mean` consistently rank highest — the RSS signal between sensors 1 and 2 carries the most activity-distinguishing information.

---

### Notebook 2 — Classification (`02_classification.ipynb`)

**Binary: bending vs. other**

1. Scatter plots confirm clean class separation in mean/max feature space.
2. Grid search over `(l, p)` via 5-fold `StratifiedKFold` selects best feature count `p` and segment depth `l`.
3. RFE + Logistic Regression fit on selected features.
4. Class imbalance (bending is minority) addressed via case-control oversampling.
5. L1-penalized LR (`LogisticRegressionCV`) compared as alternative — matches accuracy while selecting features automatically.

**Multi-class: all 7 activities**

1. L1 Multinomial Logistic Regression — best overall.
2. Gaussian Naïve Bayes — competitive; assumes feature independence.
3. Multinomial Naïve Bayes — weakest; requires non-negative inputs (MinMaxScaler applied), but the count-data assumption is a poor fit.

---

## Key observations

**Near-perfect binary separation** — bending activities are linearly separable from others in the mean/max space, which causes standard logistic regression coefficients to diverge (complete separation problem). L1 regularisation stabilises this.

**`l=1` is sufficient** — splitting time series into more segments does not improve class separability, suggesting the global summary statistics already encode the relevant motion signature.

**Feature selection methods compared** — RFE (wrapper, p-value based) vs. L1 (embedded, simultaneous selection + regularisation). L1 is preferred: single-step, auto-tuned λ, more numerically stable.

---

## Tech stack

Python · pandas · numpy · scikit-learn · scipy · matplotlib

---

## Context

DSCI 552 Homework 3 & 4 — University of Southern California  
Merged into a single project for portfolio presentation.