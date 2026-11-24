
import streamlit as st
import pandas as pd
from pathlib import Path
import time

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Wind Turbine Predictive Maintenance",
    page_icon="🌬️",
    layout="wide"
)

st.title("🌬️ Wind Turbine Predictive Maintenance")
st.caption(
    "Real-Time Predictive Maintenance Using Big Data Streaming Architecture"
)

DATA_FILE = Path(__file__).resolve().parent / "predictions.csv"

# --------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------

st.sidebar.header("Dashboard Settings")

refresh = st.sidebar.checkbox(
    "Auto refresh every 5 seconds",
    value=True
)

# --------------------------------------------------
# LOAD PREDICTION DATA
# --------------------------------------------------

if DATA_FILE.exists():

    df = pd.read_csv(DATA_FILE)

    if not df.empty:

        # Prepare prediction labels
        df["prediction"] = (
            df["prediction"].astype(str).str.lower().str.strip()
        )

        # Convert alarm values into proper Boolean values
        if "alarm" in df.columns:
            df["alarm_flag"] = (
                df["alarm"]
                .astype(str)
                .str.lower()
                .str.strip()
                .isin(["true", "1", "yes"])
            )
        else:
            df["alarm_flag"] = False

        # Sort records by timestamp
        df["parsed_time"] = pd.to_datetime(
            df["time_stamp"],
            errors="coerce"
        )

        df = df.sort_values(
            "parsed_time",
            na_position="first"
        )

        # --------------------------------------------------
        # SYSTEM OVERVIEW
        # --------------------------------------------------

        total = len(df)

        anomalies = int(
            (df["prediction"] == "anomaly").sum()
        )

        normal = int(
            (df["prediction"] == "normal").sum()
        )

        # Latest status for each turbine
        latest_records = (
            df.groupby("asset_id", sort=False)
            .tail(1)
            .copy()
        )

        active_alarm_assets = latest_records[
            latest_records["alarm_flag"]
        ]

        active_alarms = len(active_alarm_assets)

        st.subheader("📊 System Overview")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Records", total)
        col2.metric("Normal Predictions", normal)
        col3.metric("Anomalies", anomalies)
        col4.metric("Active Alarms", active_alarms)

        st.divider()

        # --------------------------------------------------
        # CRITICAL ALARM NOTIFICATIONS
        # --------------------------------------------------

        st.subheader("🚨 Critical Alarm Notifications")

        if not active_alarm_assets.empty:

            for _, row in active_alarm_assets.iterrows():

                st.error(
                    f"🚨 CRITICAL ALARM | "
                    f"Turbine Asset ID: {row['asset_id']}\n\n"
                    f"Time: {row['time_stamp']}\n\n"
                    f"Anomaly Probability: "
                    f"{row.get('anomaly_probability', 'N/A')}\n\n"
                    f"Maintenance inspection required!"
                )

        else:

            st.success(
                "No active critical alarms. "
                "All turbines are currently below the alarm condition."
            )

        st.divider()

        # --------------------------------------------------
        # TURBINE-WISE LIVE STATUS
        # --------------------------------------------------

        st.subheader("🌬️ Turbine-wise Live Status")

        turbine_status = []

        for asset_id, group in df.groupby("asset_id", sort=False):

            latest = group.iloc[-1]

            turbine_anomalies = int(
                (group["prediction"] == "anomaly").sum()
            )

            if latest["alarm_flag"]:

                status = "🚨 CRITICAL ALARM"

            elif latest["prediction"] == "anomaly":

                status = "⚠️ ANOMALY DETECTED"

            else:

                status = "🟢 NORMAL"

            turbine_status.append({
                "Asset ID": asset_id,
                "Latest Reading": latest["time_stamp"],
                "Condition": status,
                "Total Anomalies": turbine_anomalies,
                "Active Alarm": (
                    "YES" if latest["alarm_flag"] else "NO"
                )
            })

        status_df = pd.DataFrame(turbine_status)

        st.dataframe(
            status_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # --------------------------------------------------
        # PREDICTION DISTRIBUTION
        # --------------------------------------------------

        left, right = st.columns(2)

        with left:

            st.subheader("📈 Prediction Distribution")

            distribution = pd.DataFrame({
                "Status": ["Normal", "Anomaly"],
                "Count": [normal, anomalies]
            })

            st.bar_chart(
                distribution.set_index("Status")
            )

        with right:

            st.subheader("📉 Anomaly Probability")

            if "anomaly_probability" in df.columns:

                chart_data = df[
                    ["anomaly_probability"]
                ].copy()

                chart_data.index = range(1, len(chart_data) + 1)

                st.line_chart(chart_data)

        st.divider()

        # --------------------------------------------------
        # ALARM HISTORY
        # --------------------------------------------------

        
        st.subheader("📋 Alarm History")

        alarm_file = Path(__file__).parent / "alarm_history.csv"

        if alarm_file.exists():
            alarm_history = pd.read_csv(alarm_file)

            if not alarm_history.empty:
                history_columns = [
                    column for column in [
                        "asset_id",
                        "time_stamp",
                        "anomaly_probability",
                        "consecutive_anomalies",
                        "alarm",
                    ]
                    if column in alarm_history.columns
                ]

                st.dataframe(
                    alarm_history[history_columns].iloc[::-1],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No critical alarm events have been recorded yet.")
        else:
            st.info("No critical alarm events have been recorded yet.")


        # --------------------------------------------------
        # RECENT TURBINE PREDICTIONS
        # --------------------------------------------------
        st.subheader("🔍 Recent Turbine Predictions")

        display_columns = [
            column for column in [
                "asset_id",
                "time_stamp",
                "prediction",
                "anomaly_probability",
                "alarm",
                "consecutive_anomalies"
            ]
            if column in df.columns
        ]

        st.dataframe(
            df[display_columns].tail(100).iloc[::-1],
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Dashboard refreshes automatically every 5 seconds "
            "when auto-refresh is enabled."
        )

    else:

        st.info("No prediction records available yet.")

else:

    st.info(
        "Waiting for prediction data from the Spark streaming application."
    )

    st.write(
        "Start the Spark Kafka consumer to populate the dashboard."
    )

# --------------------------------------------------
# AUTO REFRESH
# --------------------------------------------------

if refresh:

    time.sleep(5)
    st.rerun()

