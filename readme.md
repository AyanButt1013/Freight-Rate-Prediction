# Freight Rate Prediction Challenge

See `Freight_Rate_ML_Assessment.pdf` for the assessment instructions.

## What to do

1. Train and validate your model using `data/train_test.csv`.
2. Predict every load in `data/validation.csv`. Each load has a unique `load_id`.
3. Fill the matching `predicted_rate` values in `data/validation_predictions_template.csv` and save it as `validation_predictions.csv`.
4. Predict every row in `data/december_chart_inputs.csv` by filling its `predicted_rate` column.
5. Install the scorer requirements and run:

```bash
python -m pip install -r requirements.txt
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The scorer validates both files and creates `scorer_results/candidate_december.png`.

## Submit

- GitHub repository containing your code, dependencies, and run instructions
- `validation_predictions.csv`
- PDF or DOCX report containing your validation, data split approach and `candidate_december.png`
- 2-3 minute Loom link
# Freight Rate Prediction using Deep Learning

A PyTorch-based freight rate prediction model developed as part of a machine learning assessment.

The model predicts the freight rate (`posted_rate`) from shipment information such as:

- Pickup & delivery coordinates
- Distance
- Weight
- Market index
- Quote signal
- Equipment type
- Date

The final model achieved:

| Metric | Value |
|---------|--------|
| MAE | **105.46** |
| RMSE | **529.54** |
| R² Score | **0.8689** |

---

# Project Structure

```
.
├── main.py
├── predictor.py
├── score.py
├── trainer.py
├── evaluator.py
├── model.py
├── data_prepare.py
├── eda.py
├── requirements.txt
├── README.md
├── freight_rate_model.pth
├── scaler.pkl
└── ...
```

---

# Features

- PyTorch Neural Network
- Feature Engineering
- Log-transformed target
- Huber Loss
- ReduceLROnPlateau scheduler
- Early Stopping
- StandardScaler preprocessing
- One-Hot Encoding
- Missing value imputation

---

# Installation

Clone the repository

```bash
git clone https://github.com/<username>/Freight-Rate-Prediction.git
cd Freight-Rate-Prediction
```

Create a virtual environment

Linux / WSL

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset

Place the following files inside the project folder.

```
train-test.csv
validation.csv
december-chart-inputs.csv
```

These datasets are **not included** in this repository.

---

# Exploratory Data Analysis

Run

```bash
python eda.py
```

This generates

- Histograms
- Correlation Matrix
- Boxplots
- Missing value analysis
- Distribution plots

---

# Training the Model

Run

```bash
python main.py
```

The training pipeline performs

- Load dataset
- Missing value imputation
- Date feature extraction
- Feature engineering
- One-hot encoding
- Feature scaling
- Train/Validation split
- Model training
- Early stopping
- Evaluation

Outputs

```
freight_rate_model.pth
scaler.pkl
```

---

# Predicting Validation Data

Run

```bash
python predict_validation.py
```

This generates

```
validation-predictions.csv
```

containing

```
load_id,predicted_rate
```

---

# Predicting December Rates

Complete

```
december-chart-inputs.csv
```

using

```python
from predictor import FreightPredictor
```

Then save the completed file.

---

# Running the Official Validator

```bash
python score.py \
    --predictions validation-predictions.csv \
    --december-predictions december-chart-inputs.csv
```

Successful execution prints

```
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart:
scorer_results/candidate_december.png
```

---

# Model Architecture

```
Input (14 features)

↓

Linear(14 → 256)

↓

ReLU

↓

Dropout(0.2)

↓

Linear(256 → 128)

↓

ReLU

↓

Dropout(0.2)

↓

Linear(128 → 64)

↓

ReLU

↓

Linear(64 → 1)
```

---

# Feature Engineering

Additional engineered features

- delta_lat
- delta_lon

Target transformation

```
log1p(posted_rate)
```

Prediction transformation

```
expm1(prediction)
```

---

# Training Configuration

| Parameter | Value |
|------------|---------|
| Optimizer | Adam |
| Loss | Huber Loss |
| Learning Rate | 0.001 |
| Scheduler | ReduceLROnPlateau |
| Early Stopping | Yes |
| Epochs | 35 |
| Batch Size | 64 |

---

# Results

| Metric | Score |
|---------|-------|
| MAE | 105.46 |
| RMSE | 529.54 |
| R² | 0.8689 |

---

# Requirements

- Python 3.11+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Joblib

Install using

```bash
pip install -r requirements.txt
```

---

# Author

Ayan Butt

Mechanical Engineer | Machine Learning Engineer
