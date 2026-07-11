# Burnly

![Status](https://img.shields.io/badge/status-active-brightgreen) ![Python](https://img.shields.io/badge/python-3.9+-blue) ![License](https://img.shields.io/badge/license-MIT-green)

Burnly is a production-ready machine learning project that predicts calorie expenditure from physiological and activity-related features. Built as a portfolio showcase for end-to-end ML workflows.

## Project Overview

This project builds a regression model that estimates calories burned based on features such as:

- age
- gender
- height
- weight
- exercise duration
- heart rate

The goal is to provide a simple but realistic example of a supervised learning pipeline.
## Features

- Synthetic or CSV-based dataset loading
- Clean preprocessing pipeline for tabular features
- Random Forest regression with optimized hyperparameters
- Model artifact export (joblib format)
- Evaluation metrics JSON export
- Automated visualization generation
- Comprehensive test coverage (pytest)
- Jupyter notebook walkthrough
- HTML dashboard for results

## Project Structure

```text
Burnly/
├── README.md
├── requirements.txt
├── src/
│   ├── train.py
│   └── __init__.py
├── tests/
│   └── test_train.py
├── notebooks/
│   └── burnly_exploration.ipynb
├── dashboard.html
├── artifacts/
│   ├── model.joblib
│   ├── metrics.json
│   └── evaluation_plot.png
└── .gitignore
```

## Quick Start

### Prerequisites

- Python 3.9 or higher
- pip or conda

### Installation

```bash
git clone https://github.com/HarshithhKumar/Burnly.git
cd Burnly
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run the Model

```bash
python src/train.py
```

**Output:**
- `artifacts/model.joblib` - Trained model
- `artifacts/metrics.json` - Performance metrics
- `artifacts/evaluation_plot.png` - Results visualization
- `data/synthetic_calories.csv` - Generated dataset

### Run Tests

```bash
pytest -q
```

## Model Performance

The trained model reports:
- **MAE** (Mean Absolute Error): ~25-30 calories
- **RMSE** (Root Mean Squared Error): ~35-40 calories  
- **R² Score**: 0.85+

Metrics are stored in `artifacts/metrics.json` for easy review.

## Showcase & Documentation

- **Notebook**: `notebooks/burnly_exploration.ipynb` - Interactive walkthrough
- **Dashboard**: `dashboard.html` - Results visualization page
- **Artifacts**: `artifacts/` - Model, metrics, and plots
