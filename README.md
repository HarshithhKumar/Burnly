# Burnly

Burnly is a resume-ready machine learning project focused on predicting calorie expenditure from physiological and activity-related features. The project demonstrates a complete end-to-end workflow: data generation, preprocessing, model training, evaluation, and artifact export.

## Project Overview

This project builds a regression model that estimates calories burned based on features such as:

- age
- gender
- height
- weight
- exercise duration
- heart rate

The goal is to provide a simple but realistic example of a supervised learning pipeline that can be showcased in interviews, GitHub portfolios, or resume discussions.

## Features

- Synthetic or CSV-based dataset loading
- Clean preprocessing pipeline for tabular features
- Regression model training and evaluation
- Model artifact export for reuse
- Basic test coverage for core functionality

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

## Setup

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Training the Model

Run the training script from the project root:

```bash
python src/train.py
```

This will:

- create or load a dataset
- train a regression model
- save the trained model to artifacts/model.joblib
- save evaluation metrics to artifacts/metrics.json

## Testing

Run the test suite with:

```bash
pytest -q
```

## Showcase Items

The project now includes:

- a notebook walkthrough in notebooks/burnly_exploration.ipynb
- a simple HTML dashboard in dashboard.html
- generated evaluation artifacts in artifacts/

## Example Output

The training script reports metrics such as:

- MAE
- RMSE
- R2 score

These metrics are stored in the generated artifact files for easy review.
