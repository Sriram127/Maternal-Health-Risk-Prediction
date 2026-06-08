from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "Maternal Health Risk Data Set.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "maternal_health_risk_model.joblib"
FEATURES = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
TARGET = "RiskLevel"
BOM_CODEPOINTS = {187, 191, 239, 65279}


def load_data():
    data = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    data.columns = [
        "".join(character for character in column if ord(character) not in BOM_CODEPOINTS).strip()
        for column in data.columns
    ]
    return data


def train_model():
    data = load_data()
    x = data[FEATURES]
    y = data[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=300, random_state=42)),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "classification_report": classification_report(y_test, predictions, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=sorted(y.unique())),
        "labels": sorted(y.unique()),
        "rows": len(data),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "metrics": metrics}, MODEL_PATH)
    return model, metrics


def load_or_train_model():
    if MODEL_PATH.exists():
        payload = joblib.load(MODEL_PATH)
        return payload["model"], payload["metrics"]
    return train_model()


def predict_risk(values):
    model, _metrics = load_or_train_model()
    frame = pd.DataFrame([values], columns=FEATURES)
    prediction = model.predict(frame)[0]
    probabilities = model.predict_proba(frame)[0]
    classes = model.named_steps["classifier"].classes_
    confidence = {
        risk_class: float(probability)
        for risk_class, probability in zip(classes, probabilities)
    }
    return prediction, confidence
