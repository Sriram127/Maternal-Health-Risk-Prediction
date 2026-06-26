# Maternal Health Risk Prediction

A machine learning web app that predicts maternal health risk levels from clinical readings using a Random Forest classifier.

> **Disclaimer:** This project is for education and portfolio demonstration only. It is not medical advice and should not be used for clinical diagnosis.

## Features

- Streamlit prediction app with dark theme
- Reproducible training pipeline (model trains automatically on first run)
- Random Forest classifier
- Validation accuracy and confusion matrix
- Class probability breakdown (low / mid / high risk)
- Dataset preview and feature explanation
- Original EDA notebook preserved

## Tech Stack

- Python 3
- Streamlit
- scikit-learn
- pandas / NumPy
- Jupyter Notebook (EDA)

## Dataset

The dataset contains these input features:

| Feature | Description |
|---|---|
| `Age` | Patient age in years |
| `SystolicBP` | Systolic blood pressure (mmHg) |
| `DiastolicBP` | Diastolic blood pressure (mmHg) |
| `BS` | Blood glucose level (mmol/L) |
| `BodyTemp` | Body temperature (°F) |
| `HeartRate` | Resting heart rate (bpm) |

Target: `RiskLevel` — `low risk`, `mid risk`, or `high risk`.

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── README.md
├── Maternal Health Risk Data Set.csv
├── maternal-health-risk-using-edaand-machine-learning.ipynb
└── src/
    ├── __init__.py
    └── model.py
```

## Setup — Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Sriram127/Maternal-Health-Risk-Prediction.git
cd Maternal-Health-Risk-Prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## How It Works

1. The app loads the maternal health dataset on startup.
2. A Random Forest classifier is trained if no saved model exists.
3. The user enters clinical readings via sidebar sliders.
4. The app displays the predicted risk class and per-class probabilities.

## Future Improvements

- Compare multiple models (Gradient Boost, XGBoost, Logistic Regression) and save the best one
- Add cross-validation and ROC curve visualisations
- Add SHAP feature importance explanations
