# Movie Rating Prediction Model

**Authors:** Tiferet Bluka (325204113) & Ziv Nagad (322558271)

---

## Overview

A machine learning model that predicts the average IMDb rating of a movie **before its release**, using only features available prior to the release date (genre, release year, runtime, country of origin, language).

The final model saved in `model.pkl` is **ElasticNet** (alpha=0.0005, l1_ratio=0.15), as required for the competition evaluation.

---

## File Structure

```
├── movie_rating_model.ipynb   # Full Python notebook (EDA, training, evaluation)
├── model.pkl                  # Trained ElasticNet model
├── report.pdf                 # Full project report
├── requirements.txt           # Required libraries
└── README.md                  # This file
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the notebook
```bash
jupyter notebook movie_rating_model.ipynb
```
Make sure `dataset.csv` is in the same directory.

### 3. Load model and predict
```python
import joblib
import pandas as pd

# Copy prepare_data function from notebook
df = pd.read_csv('dataset.csv')
X = prepare_data(df)

model = joblib.load('model.pkl')
y_pred = model.predict(X)
```

---

## Results (10-Fold Cross Validation)

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| ElasticNet | 1.1392 ± 0.0139 | 0.8735 ± 0.0091 | 0.2252 ± 0.0066 |
| Random Forest | 1.1115 ± 0.0128 | 0.8481 ± 0.0089 | 0.2624 ± 0.0058 |

---

## Feature Engineering

| Feature | Type | Description |
|---------|------|-------------|
| `movie_age` | Numeric | 2025 - release year |
| `num_genres` | Numeric | Number of genres |
| `runtime_per_genre` | Numeric (Complex) | runtimeMinutes / num_genres |
| `runtime_category` | Categorical → OHE | short / medium / long |
| `is_english` | Binary | 1 if English in languages |
| `is_usa` | Binary | 1 if United States in country |
| `is_europe` | Binary | 1 if France/Germany/Italy/Spain |
| `is_india` | Binary | 1 if India in country |
| `is_east_asian` | Binary | 1 if Japan/South Korea/China |
| `genre_*` | Binary ×8 | Drama, Comedy, Documentary, Horror, Action, Romance, Thriller, Crime |

---

## Technical Requirements

- Python 3.10+
- See `requirements.txt` for full list of dependencies
- `random_state=42` used throughout for reproducibility
