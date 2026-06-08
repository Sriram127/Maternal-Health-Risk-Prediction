# Maternal Health Risk Prediction

A machine learning project that predicts maternal health risk levels from clinical readings.

> This project is for education and portfolio demonstration only. It is not medical advice and should not be used for diagnosis.

## Features

- Streamlit prediction app
- Reproducible training pipeline
- Random Forest classifier
- Validation accuracy and confusion matrix
- Dataset preview and feature explanation
- Original EDA notebook preserved

## Dataset

The dataset contains these input features:

- `Age`
- `SystolicBP`
- `DiastolicBP`
- `BS`
- `BodyTemp`
- `HeartRate`

Target column:

- `RiskLevel`: low risk, mid risk, or high risk

## Project Structure

```text
.
|-- app.py
|-- requirements.txt
|-- README.md
|-- Maternal Health Risk Data Set.csv
|-- maternal-health-risk-using-edaand-machine-learning.ipynb
`-- src/
    |-- __init__.py
    `-- model.py
```

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How It Works

1. The app loads the maternal health dataset.
2. A Random Forest classifier is trained if no saved model exists.
3. The user enters clinical readings.
4. The app predicts the maternal health risk class and displays class probabilities.

## Future Improvements

- Compare multiple models and save the best one.
- Add cross-validation.
- Add SHAP or feature importance explanations.
- Deploy on Streamlit Community Cloud.
