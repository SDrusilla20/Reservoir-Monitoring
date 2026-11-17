
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import time
import tensorflow as tf

# ---------------- CONFIGURATION ----------------

st.set_page_config(
    page_title="Reservoir AI Monitoring",
    page_icon="💧",
    layout="wide"
)

BASE = ""

PRIMARY = [
    "level_ft",
    "inflow_cusecs",
    "outflow_cusecs",
    "storage_mcft"
]

# ---------------- LOAD ARTIFACTS ----------------

@st.cache_resource
def load_artifacts():
    isolation_forest = joblib.load(BASE + "isolation_forest.pkl")
    autoencoder = tf.keras.models.load_model(
        BASE + "optimized_autoencoder.keras"
    )
    scaler = joblib.load(BASE + "scaler.pkl")

    with open(BASE + "thresholds.json") as f:
        thresholds = json.load(f)

    with open(BASE + "features.json") as f:
        features = json.load(f)

    return isolation_forest, autoencoder, scaler, thresholds, features


@st.cache_data
def load_data():
    df = pd.read_csv(BASE + "reservoir_data.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


try:
    isolation_forest, autoencoder, scaler, thresholds, features = load_artifacts()
    df = load_data()
except Exception as e:
    st.error(f"Could not load project files: {e}")
    st.stop()


# ---------------- FEATURE ENGINEERING ----------------

def create_features(data):
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)

    for col in PRIMARY:
        data[f"{col}_change"] = data[col].diff()

    data["net_flow"] = (
        data["inflow_cusecs"] - data["outflow_cusecs"]
    )

    data = data.set_index("date")

    for col in PRIMARY:
        previous = data[col].shift(1)

        data[f"{col}_rolling_mean_14d"] = (
            previous.rolling("14D", min_periods=5).mean()
        )

        data[f"{col}_rolling_std_14d"] = (
            previous.rolling("14D", min_periods=5).std()
        )

    return data.reset_index()


# ---------------- MODEL PREDICTION ----------------

def detect_anomalies(data, selected_model):
    data = data.copy()

    data = data.dropna(subset=features).copy()

    X = scaler.transform(data[features])

    if selected_model == "Isolation Forest":
        scores = -isolation_forest.decision_function(X)
        threshold = thresholds["isolation_forest"]

    else:
        reconstructed = autoencoder.predict(X, verbose=0)
        scores = np.mean(np.square(X - reconstructed), axis=1)
        threshold = thresholds["autoencoder"]

    data["anomaly_score"] = scores
    data["prediction"] = (scores >= threshold).astype(int)

    return data, threshold


# ---------------- SIDEBAR ----------------

st.sidebar.title("💧 Reservoir Monitoring")

selected_model = st.sidebar.selectbox(
    "Select anomaly detection model",
    ["Isolation Forest", "Optimized Autoencoder"]
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Historical Monitoring",
        "Anomaly Alerts",
        "Model Comparison",
        "Live Simulation"
    ]
)

# ---------------- PROCESS DATA ----------------

featured_data = create_features(df)

try:
    results, threshold = detect_anomalies(
        featured_data,
        selected_model
    )
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# ---------------- HEADER ----------------

st.title("💧 AI-Based Reservoir Monitoring System")

st.caption(
    "Historical reservoir monitoring | Unsupervised anomaly detection"
)

# ---------------- OVERVIEW ----------------

if page == "Overview":

    st.subheader("Reservoir Overview")

    latest = results.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Water Level", f"{latest.level_ft:.2f} ft")
    c2.metric("Inflow", f"{latest.inflow_cusecs:.2f} cusecs")
    c3.metric("Outflow", f"{latest.outflow_cusecs:.2f} cusecs")
    c4.metric("Storage", f"{latest.storage_mcft:.2f} MCFT")

    normal_count = (results["prediction"] == 0).sum()
    anomaly_count = (results["prediction"] == 1).sum()

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Observations", len(results))
    c2.metric("Normal Observations", normal_count)
    c3.metric("Flagged Observations", anomaly_count)

    st.subheader("Reservoir Variable Trends")

    for col, label in zip(
        PRIMARY,
        ["Water Level (ft)", "Inflow (cusecs)",
         "Outflow (cusecs)", "Storage (MCFT)"]
    ):
        st.markdown(f"**{label}**")
        st.line_chart(results.set_index("date")[col])

    st.subheader("Recent Anomaly Alerts")

    st.dataframe(
        results[results["prediction"] == 1][
            ["date"] + PRIMARY + ["anomaly_score"]
        ].tail(10),
        use_container_width=True
    )


# ---------------- HISTORICAL MONITORING ----------------

elif page == "Historical Monitoring":

    st.subheader("Historical Reservoir Readings")

    st.line_chart(
        results.set_index("date")[PRIMARY]
    )

    st.subheader("Anomaly Scores")

    score_data = results.set_index("date")[["anomaly_score"]]
    st.line_chart(score_data)

    st.caption(f"Detection threshold: {threshold:.6f}")


# ---------------- ANOMALY ALERTS ----------------

elif page == "Anomaly Alerts":

    st.subheader("Detected Anomalies")

    alerts = results[results["prediction"] == 1].copy()

    st.metric("Total Flagged Observations", len(alerts))

    if len(alerts):
        alerts["alert_status"] = "Potential anomaly"

        st.dataframe(
            alerts[
                ["date"] + PRIMARY +
                ["anomaly_score", "alert_status"]
            ].sort_values("anomaly_score", ascending=False),
            use_container_width=True
        )

        st.download_button(
            "Download anomaly report",
            alerts.to_csv(index=False),
            file_name="reservoir_anomaly_alerts.csv",
            mime="text/csv"
        )
    else:
        st.success("No observations crossed the selected threshold.")


# ---------------- MODEL COMPARISON ----------------

elif page == "Model Comparison":

    st.subheader("Model Performance Comparison")

    try:
        comparison = pd.read_csv(BASE + "model_comparison.csv")

        st.dataframe(
            comparison,
            use_container_width=True
        )

        st.bar_chart(
            comparison.pivot(
                index="Scenario",
                columns="Model",
                values="PR-AUC"
            )
        )

    except Exception as e:
        st.error(f"Could not load comparison results: {e}")


# ---------------- LIVE SIMULATION ----------------

elif page == "Live Simulation":

    st.subheader("Historical Data Replay Simulation")

    st.info(
        "This mode replays historical daily observations one at a time. "
        "It is a software simulation, not a live ESP32 sensor connection."
    )

    speed = st.slider(
        "Replay interval (seconds)",
        min_value=1,
        max_value=5,
        value=1
    )

    if "replay_index" not in st.session_state:
        st.session_state.replay_index = 0

    total = len(results)

    col1, col2 = st.columns(2)

    if col1.button("Start / Next Reading"):
        st.session_state.replay_index = min(
            st.session_state.replay_index + 1,
            total
        )

    if col2.button("Reset Replay"):
        st.session_state.replay_index = 0

    if st.session_state.replay_index > 0:

        current = results.iloc[
            st.session_state.replay_index - 1
        ]

        st.markdown(f"### Reading Date: {current.date}")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Water Level", f"{current.level_ft:.2f} ft")
        c2.metric("Inflow", f"{current.inflow_cusecs:.2f}")
        c3.metric("Outflow", f"{current.outflow_cusecs:.2f}")
        c4.metric("Storage", f"{current.storage_mcft:.2f}")

        if current.prediction == 1:
            st.error(
                f"Potential anomaly detected! "
                f"Score: {current.anomaly_score:.5f}"
            )
        else:
            st.success("Reading classified as normal.")

        st.progress(
            st.session_state.replay_index / total
        )

        st.caption(
            f"Reading {st.session_state.replay_index} of {total}"
        )

    else:
        st.info("Press Start / Next Reading to begin replay.")

st.sidebar.divider()
st.sidebar.caption("AI Reservoir Monitoring | Academic Project")
