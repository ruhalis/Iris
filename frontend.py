import os
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Iris Classifier", page_icon="🌸")
st.title("Iris Classifier")

try:
    info = requests.get(f"{API_URL}/info", timeout=5).json()
    c1, c2 = st.columns(2)
    c1.metric("Model", info.get("model_name", "unknown"))
    f1 = info.get("f1_macro")
    c2.metric("F1 (macro)", f"{f1:.4f}" if isinstance(f1, (int, float)) else "n/a")
except Exception as e:
    st.warning(f"Could not load model info: {e}")

st.divider()

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.number_input("Sepal length (cm)", 0.0, 10.0, 5.1, 0.1)
    sepal_width = st.number_input("Sepal width (cm)", 0.0, 10.0, 3.5, 0.1)
with col2:
    petal_length = st.number_input("Petal length (cm)", 0.0, 10.0, 1.4, 0.1)
    petal_width = st.number_input("Petal width (cm)", 0.0, 10.0, 0.2, 0.1)

if st.button("Predict", type="primary"):
    payload = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        r.raise_for_status()
        result = r.json()
        st.success(f"Predicted: **{result['predicted_label']}** (class {result['predicted_class']})")
        st.subheader("Probabilities")
        st.bar_chart(result["probabilities"])
        st.json(result)
    except Exception as e:
        st.error(f"Request failed: {e}")
