
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
    scaler = joblib.load(BASE + "scaler.pkl")

    with open(BASE + "thresholds.json") as f:
        thresholds = json.load(f)

    with open(BASE + "features.json") as f:
        features = json.load(f)

    return isolation_forest, scaler, thresholds, features


@st.cache_data
def load_data():
    df = pd.read_csv(BASE + "reservoir_replaydata.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


try:
    isolation_forest, scaler, thresholds, features = load_artifacts()
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

def detect_anomalies(data):
    data = data.copy()
    data = data.dropna(subset=features).copy()

    X = scaler.transform(data[features])

    scores = -isolation_forest.decision_function(X)
    threshold = thresholds["isolation_forest"]

    data["anomaly_score"] = scores
    data["prediction"] = (scores >= threshold).astype(int)

    return data, threshold


# ---------------- SIDEBAR ----------------

st.sidebar.title("💧 Reservoir Monitoring")

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
    results, threshold = detect_anomalies(featured_data)
    st.write("### Anomaly Detection Diagnostics")

    st.write("Replay dataset rows:", len(df))
    st.write("Valid rows after feature engineering:", len(results))
    st.write("Detection threshold:", threshold)

    st.write(
        "Maximum anomaly score:",
        results["anomaly_score"].max()
    )

    st.write(
        "Minimum anomaly score:",
        results["anomaly_score"].min()
    )

    st.write(
        "Predictions:",
        results["prediction"].value_counts().to_dict()
    )
    
    if "injected_anomaly" in results.columns:
        st.write(
            "Injected anomalies remaining:",
            int(results["injected_anomaly"].sum())
        )
    
        st.write(
            "Injected anomalies detected:",
            int(
                (
                    (results["injected_anomaly"] == 1)
                    & (results["prediction"] == 1)
                ).sum()
            )
        )
    
        st.write(
            "Injected anomalies missed:",
            int(
                (
                    (results["injected_anomaly"] == 1)
                    & (results["prediction"] == 0)
                ).sum()
            )
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

    st.subheader("Automatic Reservoir Reading Replay")

    st.info(
        "Historical daily observations and injected synthetic "
        "contextual anomalies are replayed automatically. "
        "This is a software simulation, not a live ESP32 connection."
    )

    speed = st.slider(
        "Time between readings (seconds)",
        min_value=1,
        max_value=5,
        value=2
    )

    if "replay_index" not in st.session_state:
        st.session_state.replay_index = 0

    if "replay_running" not in st.session_state:
        st.session_state.replay_running = False

    total = len(results)

    col1, col2, col3 = st.columns(3)

    if col1.button("Start Replay"):
        st.session_state.replay_running = True

    if col2.button("Stop Replay"):
        st.session_state.replay_running = False

    if col3.button("Reset Replay"):
        st.session_state.replay_running = False
        st.session_state.replay_index = 0

    @st.fragment(
        run_every=speed if st.session_state.replay_running else None
    )
    def automatic_replay():

        if st.session_state.replay_index < total:

            current = results.iloc[
                st.session_state.replay_index
            ]

            st.markdown(f"### Reading Date: {current['date']}")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Water Level", f"{current['level_ft']:.2f} ft")
            c2.metric("Inflow", f"{current['inflow_cusecs']:.2f} cusecs")
            c3.metric("Outflow", f"{current['outflow_cusecs']:.2f} cusecs")
            c4.metric("Storage", f"{current['storage_mcft']:.2f} MCFT")

            if current["prediction"] == 1:
                st.error(
                    f"ANOMALY DETECTED | "
                    f"Score: {current['anomaly_score']:.5f}"
                )
            else:
                st.success("Reading classified as normal.")

            st.progress(
                (st.session_state.replay_index + 1) / total
            )

            st.caption(
                f"Reading {st.session_state.replay_index + 1} of {total}"
            )

            st.session_state.replay_index += 1

        else:
            st.session_state.replay_running = False
            st.success("Replay completed.")
            st.rerun()

    automatic_replay()

st.sidebar.divider()
st.sidebar.caption("AI Reservoir Monitoring")
