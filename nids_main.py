import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="AI-Based NIDS", layout="wide")

st.title("🔐 AI-Based Network Intrusion Detection System")
st.markdown("Final Year Project using Machine Learning (Random Forest)")

def load_training_data():
    train_df = pd.read_csv("Train_data.csv")

    possible_labels = ['label', 'Label', 'class', 'attack', 'target', 'outcome']
    label_col = None

    for col in possible_labels:
        if col in train_df.columns:
            label_col = col
            break

    if label_col is None:
        st.error("❌ Label column not found in training dataset")
        st.stop()

    train_df[label_col] = train_df[label_col].apply(
        lambda x: 0 if str(x).lower() in ['normal', '0', 'benign'] else 1
    )

    features = train_df.select_dtypes(include=np.number).columns.tolist()
    features.remove(label_col)

    X = train_df[features]
    y = train_df[label_col]

    return X, y, features

st.subheader("📊 Training Dataset Preview")
st.dataframe(pd.read_csv("Train_data.csv").head())

st.sidebar.header("⚙️ Model Control")

if st.sidebar.button("Train Model Now"):
    X, y, features = load_training_data()

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)

    st.success("✅ Model Trained Successfully")
    st.info(f"🎯 Training Accuracy: {acc:.2f}")

    st.subheader("📄 Classification Report (Training Data)")
    st.text(classification_report(y, y_pred))

    st.session_state["model"] = model
    st.session_state["features"] = features

st.subheader("🚦 Live Intrusion Detection")

if "model" in st.session_state:
    user_input = []

    for feature in st.session_state["features"]:
        value = st.number_input(f"{feature}", value=0.0)
        user_input.append(value)

    if st.button("Detect Intrusion"):
        prediction = st.session_state["model"].predict([user_input])

        if prediction[0] == 0:
            st.success("✅ Normal Traffic")
        else:
            st.error("🚨 Intrusion Detected!")
else:
    st.warning("⚠️ Train the model first")

st.markdown("---")
st.markdown("🎓 Final Year Project | AI-Based Network Intrusion Detection System")
