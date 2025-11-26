import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.set_page_config(page_title="Health Status Dashboard", layout="wide")

FEATURES = [
    "Physical_Activity",
    "Nutrition_Score",
    "Stress_Level",
    "Mindfulness",
    "Sleep_Hours",
    "Hydration",
    "BMI",
    "Alcohol",
    "Smoking",
    
]

# ------------------------------------------------------------
# Load model (No scaler needed)
# ------------------------------------------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


# ------------------------------------------------------------
# Session State
# ------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = pd.DataFrame(columns=FEATURES + ["Predicted_Health_Status"])

if "last_input" not in st.session_state:
    st.session_state["last_input"] = None

if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None


# ------------------------------------------------------------
# Recommendations
# ------------------------------------------------------------
def get_recommendations(d, pred):
    recs = [f"Prediction: {pred}"]

    if d["Physical_Activity"] < 30:
        recs.append("Increase physical activity.")

    if d["Nutrition_Score"] < 6:
        recs.append("Improve nutrition quality.")

    if d["Stress_Level"] > 6:
        recs.append("Reduce stress using mindfulness techniques.")

    if d["Sleep_Hours"] < 7:
        recs.append("Increase sleep to 7–9 hours.")

    if d["Hydration"] < 3:
        recs.append("Increase water intake.")

    if d["BMI"] > 25:
        recs.append("Focus on gradual weight reduction.")

    if d["Alcohol"] > 2:
        recs.append("Reduce alcohol intake.")

    if d["Smoking"] > 0:
        recs.append("Reduce or quit smoking.")


    return recs


# ------------------------------------------------------------
# Navigation
# ------------------------------------------------------------
page = st.sidebar.radio("Navigation", ["Model Prediction", "Recommendations"])


# ------------------------------------------------------------
# PAGE 1 — Prediction
# ------------------------------------------------------------
if page == "Model Prediction":
    st.title("Health Status Prediction")

    cols = st.columns(3)
    inputs = {}

    sliders = {
        "Physical_Activity": (0, 100, 40),
        "Nutrition_Score": (0, 10, 6),
        "Stress_Level": (0, 10, 5),
        "Mindfulness": (0, 30, 15),
        "Sleep_Hours": (0, 12, 7),
        "Hydration": (0.0, 6.0, 2.5),
        "BMI": (10, 40, 24),
        "Alcohol": (0, 15, 2),
        "Smoking": (0, 15, 0),
      
    }

    for i, (key, (low, high, default)) in enumerate(sliders.items()):
        with cols[i % 3]:
            if isinstance(low, float):
                inputs[key] = st.slider(key, low, high, default, step=0.1)
            else:
                inputs[key] = st.slider(key, low, high, default)

    if st.button("Predict"):
        X = np.array([list(inputs.values())])  # No scaling needed

        pred = model.predict(X)[0]

        st.success(f"Predicted Health Status: {pred}")

        st.session_state["last_input"] = inputs
        st.session_state["last_prediction"] = pred

        new_row = inputs.copy()
        new_row["Predicted_Health_Status"] = pred
        st.session_state["history"] = pd.concat(
            [st.session_state["history"], pd.DataFrame([new_row])],
            ignore_index=True,
        )

        st.write(pd.DataFrame([inputs]))


# ------------------------------------------------------------
# PAGE 2 — Recommendations
# ------------------------------------------------------------
elif page == "Recommendations":
    st.title("Recommendations")

    if st.session_state["last_input"] is None:
        st.warning("Run a prediction first.")
    else:
        d = st.session_state["last_input"]
        pred = st.session_state["last_prediction"]

        st.subheader("Last Prediction")
        st.write(pred)

        st.subheader("Recommendations")
        for r in get_recommendations(d, pred):
            st.markdown(f"- {r}")


# ------------------------------------------------------------
# PAGE 3 — History
# ------------------------------------------------------------

