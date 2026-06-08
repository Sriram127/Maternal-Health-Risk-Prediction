import matplotlib.pyplot as plt
import streamlit as st

from src.model import FEATURES, load_data, load_or_train_model, predict_risk


st.set_page_config(page_title="Maternal Health Risk Prediction", page_icon="🤰", layout="wide")

st.title("Maternal Health Risk Prediction")
st.caption("Estimate maternal health risk from clinical readings using a machine learning model.")

model, metrics = load_or_train_model()
data = load_data()

prediction_tab, metrics_tab, data_tab = st.tabs(["Prediction", "Model Metrics", "Dataset"])

with prediction_tab:
    st.warning("This tool is for educational purposes only and is not medical advice.")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=10, max_value=80, value=30)
        systolic = st.number_input("Systolic BP", min_value=60, max_value=200, value=120)
    with col2:
        diastolic = st.number_input("Diastolic BP", min_value=40, max_value=140, value=80)
        blood_sugar = st.number_input("Blood Sugar", min_value=1.0, max_value=25.0, value=7.0, step=0.1)
    with col3:
        body_temp = st.number_input("Body Temperature", min_value=90.0, max_value=110.0, value=98.0, step=0.1)
        heart_rate = st.number_input("Heart Rate", min_value=40, max_value=160, value=75)

    values = [age, systolic, diastolic, blood_sugar, body_temp, heart_rate]

    if st.button("Predict Risk", type="primary"):
        prediction, confidence = predict_risk(values)
        st.metric("Predicted risk level", prediction.title())
        st.write("Class probabilities")
        for label, probability in sorted(confidence.items()):
            st.progress(probability, text=f"{label.title()} - {probability:.1%}")

with metrics_tab:
    st.metric("Validation accuracy", f"{metrics['accuracy']:.2%}")
    st.metric("Dataset rows", metrics["rows"])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(metrics["confusion_matrix"], cmap="Purples")
    ax.set_xticks(range(len(metrics["labels"])), metrics["labels"], rotation=20)
    ax.set_yticks(range(len(metrics["labels"])), metrics["labels"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

with data_tab:
    st.write("Feature columns:", ", ".join(FEATURES))
    st.dataframe(data.head(20), use_container_width=True)
