# ML-projects

A collection of end-to-end machine learning projects organised by business problem.  
Each subdirectory is a self-contained project with its own README, notebooks, and dependencies.

---

## Projects

| Project | Problem | Tech highlights |
|---------|---------|-----------------|
| [la-crime-risk-prediction](./la-crime-risk-prediction/) | Forecast high-risk crime zones in Los Angeles by location and time | XGBoost · SHAP · H3 hexagonal grids · multi-source ETL · TimeSeriesCV |
| [human-activity-recognition](./human-activity-recognition/) | Classify physical activities from wireless RSS sensor signals | Feature engineering · L1 LR · Naïve Bayes · bootstrap CI |

---

## Structure

```
ml-projects/
├── la-crime-risk-prediction/
│   ├── src/
│   │   ├── config.py
│   │   ├── preprocess.py
│   │   ├── features.py
│   │   ├── clustering.py
│   │   ├── modeling.py
│   │   ├── explain.py
│   │   └── h3map.py
│   ├── pipeline.ipynb
│   ├── requirements.txt
│   └── README.md
│
└── human-activity-recognition/
    ├── 01_feature_engineering.ipynb
    ├── 02_classification.ipynb
    ├── requirements.txt
    └── README.md
```

---
