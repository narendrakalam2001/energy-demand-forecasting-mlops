# ============================================================
# MONITORING DASHBOARD — Energy Demand Forecasting
# ============================================================

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import numpy as np

st.set_page_config(page_title="Energy Demand Dashboard", layout="wide")
st.title("⚡ Energy Demand Forecasting — Grid Operations Dashboard")

# ── API URL ───────────────────────────────────────────────────
API_URL = os.getenv(
    "ENERGY_API_URL",
    "http://localhost:8000"
) + "/predict"

# ── PSI thresholds ────────────────────────────────────────────
PSI_MODERATE     = 0.10
PSI_HIGH         = 0.20
MAPE_ALERT       = 10.0    # %
RMSE_ALERT       = 3000.0  # MW

# ── Paths ─────────────────────────────────────────────────────
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR    = os.path.dirname(_SCRIPT_DIR)
except Exception:
    BASE_DIR = os.getcwd()

MONITOR_PATH    = os.path.join(BASE_DIR, "energy_models", "monitor_scores.csv")
LOG_PATH        = os.path.join(BASE_DIR, "logs", "prediction_logs.csv")
PSI_PATH        = os.path.join(BASE_DIR, "energy_models", "feature_drift_report.csv")
CHALLENGER_PATH = os.path.join(BASE_DIR, "energy_models", "challenger_log.json")


# ============================================================
# SIDEBAR — LIVE FORECAST
# ============================================================

st.sidebar.header("🔮 Forecast Demand (Single Hour)")

hour         = st.sidebar.slider("Hour of Day",     0, 23, 18)
day_of_week  = st.sidebar.selectbox(
    "Day of Week", [0,1,2,3,4,5,6],
    format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x]
)
month        = st.sidebar.slider("Month", 1, 12, 7)
is_weekend   = 1 if day_of_week >= 5 else 0

st.sidebar.markdown("**Lag Demand (MW)**")
demand_lag_1h   = st.sidebar.number_input("Demand 1h ago (MW)",   value=35000.0, step=100.0)
demand_lag_24h  = st.sidebar.number_input("Demand 24h ago (MW)",  value=33000.0, step=100.0)
demand_lag_168h = st.sidebar.number_input("Demand 168h ago (MW)", value=34000.0, step=100.0)

demand_roll_mean_24h = st.sidebar.number_input("Roll Mean 24h (MW)", value=34500.0, step=100.0)
demand_roll_std_24h  = st.sidebar.number_input("Roll Std 24h (MW)",  value=2000.0,  step=100.0)

temp_celsius = st.sidebar.number_input("Temperature (°C)", value=28.0)

if st.sidebar.button("⚡ Forecast Demand"):
    payload = {
        "hour":                  hour,
        "day_of_week":           day_of_week,
        "month":                 month,
        "is_weekend":            is_weekend,
        "demand_lag_1h":         demand_lag_1h,
        "demand_lag_24h":        demand_lag_24h,
        "demand_lag_168h":       demand_lag_168h,
        "demand_roll_mean_24h":  demand_roll_mean_24h,
        "demand_roll_std_24h":   demand_roll_std_24h,
        "temp_celsius":          temp_celsius,
    }
    with st.sidebar:
        with st.spinner("Calling API..."):
            try:
                response = requests.post(API_URL, json=payload, timeout=30)
                if response.status_code == 200:
                    result  = response.json()
                    action  = result["dispatch_action"]
                    color_map = {
                        "NORMAL_OPERATION":     "green",
                        "MONITOR_HIGH":         "orange",
                        "ACTIVATE_RESERVE":     "red",
                        "REDUCE_GENERATION":    "blue",
                        "EMERGENCY_PROCUREMENT":"darkred"
                    }
                    color = color_map.get(action, "gray")
                    st.success("Forecast received!")
                    st.metric("Forecast Demand", f"{result['forecast_mw']:,.0f} MW")
                    st.markdown(f"**Demand Band:** `{result['demand_band']}`")
                    st.markdown(
                        f"<h3 style='color:{color}'>Action: {action}</h3>",
                        unsafe_allow_html=True
                    )
                    if result.get("rule_triggered"):
                        st.warning(f"Rule triggered: {result['rule_triggered']}")
                else:
                    st.error(f"API error: HTTP {response.status_code}")
                    st.code(response.text[:300])
            except Exception as e:
                st.error(f"Connection error: {e}")


# ============================================================
# SECTION 1 — REAL-TIME MONITORING ALERTS
# ============================================================

st.markdown("---")
st.subheader("🚨 Real-Time Monitoring Alerts")

alerts_found = False

if os.path.exists(MONITOR_PATH):
    df_monitor = pd.read_csv(MONITOR_PATH)

    if "error_pct" in df_monitor.columns:
        avg_mape = df_monitor["error_pct"].mean()
        if avg_mape > MAPE_ALERT:
            st.error(f"🔴 HIGH MAPE: avg={avg_mape:.2f}% (threshold {MAPE_ALERT}%). Model may need retraining.")
            alerts_found = True

    if "error_mw" in df_monitor.columns:
        avg_rmse = float(np.sqrt((df_monitor["error_mw"] ** 2).mean()))
        if avg_rmse > RMSE_ALERT:
            st.warning(f"🟡 HIGH RMSE: {avg_rmse:.0f} MW (threshold {RMSE_ALERT:.0f} MW).")
            alerts_found = True

    if "decision" in df_monitor.columns:
        emergency_rate = df_monitor["decision"].str.contains("EMERGENCY").mean()
        if emergency_rate > 0.05:
            st.error(f"🔴 HIGH EMERGENCY RATE: {emergency_rate:.1%} (expected < 5%). Grid stress detected.")
            alerts_found = True

        reserve_rate = df_monitor["decision"].str.contains("ACTIVATE_RESERVE").mean()
        if reserve_rate > 0.20:
            st.warning(f"🟡 HIGH RESERVE ACTIVATION: {reserve_rate:.1%} (expected < 20%). Sustained peak demand.")
            alerts_found = True

if os.path.exists(PSI_PATH):
    df_psi_alert = pd.read_csv(PSI_PATH)
    if "drift_score" in df_psi_alert.columns:
        max_psi     = df_psi_alert["drift_score"].max()
        top_feature = df_psi_alert.iloc[0]["feature"] if "feature" in df_psi_alert.columns else "unknown"
        if max_psi >= PSI_HIGH:
            st.error(f"🔴 CRITICAL DRIFT: PSI={max_psi:.4f} on '{top_feature}'. Retrain now.")
            alerts_found = True
        elif max_psi >= PSI_MODERATE:
            st.warning(f"🟡 MODERATE DRIFT: PSI={max_psi:.4f} on '{top_feature}'. Monitor closely.")
            alerts_found = True

if not alerts_found:
    st.success("✅ All systems normal — no alerts triggered")


# ============================================================
# SECTION 2 — CHAMPION vs CHALLENGER HISTORY
# ============================================================

st.markdown("---")
st.subheader("🏆 Champion vs Challenger History")

if os.path.exists(CHALLENGER_PATH):
    with open(CHALLENGER_PATH) as f:
        challenger_log = json.load(f)

    if challenger_log:
        latest         = challenger_log[-1]
        decision_color = "green" if latest["decision"] == "PROMOTED" else "red"
        icon           = "✅" if latest["decision"] == "PROMOTED" else "❌"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Latest Challenger", latest.get("challenger_name", "—"))
        with col2:
            chall_rmse = latest.get("challenger_rmse", "—")
            st.metric("Challenger RMSE", f"{chall_rmse:.2f} MW" if isinstance(chall_rmse, float) else chall_rmse)
        with col3:
            champ_rmse = latest.get("champion_rmse", "—")
            st.metric(
                "Champion RMSE",
                f"{champ_rmse:.2f} MW" if isinstance(champ_rmse, float) else "First Run"
            )

        st.markdown(
            f"<h4 style='color:{decision_color}'>{icon} Decision: {latest['decision']} — {latest.get('reason', '')}</h4>",
            unsafe_allow_html=True
        )

        # ── 3-Gate Visual ─────────────────────────────────────
        # Gate keys match model_loader.py run_challenger_comparison():
        #   rmse_improvement_passed, r2_passed, gap_passed
        if latest.get("gates"):
            g = latest["gates"]
            gcol1, gcol2, gcol3 = st.columns(3)
            gcol1.metric("RMSE Gate", "✅ Pass" if g.get("rmse_improvement_passed") else "❌ Fail")
            gcol2.metric("R² Gate",   "✅ Pass" if g.get("r2_passed")               else "❌ Fail")
            gcol3.metric("Gap Gate",  "✅ Pass" if g.get("gap_passed")              else "❌ Fail")

        if len(challenger_log) > 1:
            with st.expander("View full challenger history"):
                history_df   = pd.DataFrame(challenger_log)
                display_cols = [c for c in [
                    "evaluated_at", "challenger_name", "challenger_rmse",
                    "champion_name", "champion_rmse", "decision", "reason"
                ] if c in history_df.columns]
                st.dataframe(history_df[display_cols])
else:
    st.info("No challenger log found. Run train_model.py to generate.")


# ============================================================
# SECTION 3 — KPI METRICS + CHARTS
# ============================================================

st.markdown("---")
st.subheader("📊 Forecast Performance KPIs")

if os.path.exists(MONITOR_PATH):
    df = pd.read_csv(MONITOR_PATH)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if "error_mw" in df.columns:
            rmse_val = float(np.sqrt((df["error_mw"] ** 2).mean()))
            st.metric("RMSE", f"{rmse_val:,.0f} MW")
    with col2:
        if "error_pct" in df.columns:
            st.metric("Avg MAPE", f"{df['error_pct'].mean():.2f}%")
    with col3:
        if "forecast" in df.columns:
            st.metric("Avg Forecast", f"{df['forecast'].mean():,.0f} MW")
    with col4:
        if "decision" in df.columns:
            normal_rate = (df["decision"] == "NORMAL_OPERATION").mean()
            st.metric("Normal Operation Rate", f"{normal_rate:.1%}")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Forecast vs Actual")
        if "forecast" in df.columns and "actual" in df.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            plot_n  = min(200, len(df))
            ax.plot(df["actual"].values[:plot_n],   label="Actual",   alpha=0.8, linewidth=1)
            ax.plot(df["forecast"].values[:plot_n], label="Forecast", alpha=0.8, linewidth=1, linestyle="--")
            ax.set_ylabel("Demand (MW)")
            ax.set_title(f"Last {plot_n} hours")
            ax.legend(fontsize=8)
            st.pyplot(fig)
            plt.close(fig)

    with col_b:
        st.subheader("Dispatch Decision Distribution")
        if "decision" in df.columns:
            counts = df["decision"].value_counts()
            fig, ax = plt.subplots(figsize=(7, 4))
            colors = {
                "NORMAL_OPERATION":     "#2ECC71",
                "MONITOR_HIGH":         "#F39C12",
                "ACTIVATE_RESERVE":     "#E74C3C",
                "REDUCE_GENERATION":    "#3498DB",
                "EMERGENCY_PROCUREMENT":"#8E44AD"
            }
            bar_colors = [colors.get(c, "gray") for c in counts.index]
            ax.bar(counts.index, counts.values, color=bar_colors)
            ax.set_ylabel("Count")
            ax.set_title("Dispatch Decisions")
            plt.xticks(rotation=20, ha="right", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    st.subheader("Demand Band Breakdown")
    if "demand_band" in df.columns:
        band_counts = df["demand_band"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 3))
        band_colors = {"LOW": "#3498DB", "NORMAL": "#2ECC71", "HIGH": "#F39C12", "PEAK": "#E74C3C"}
        ax.bar(band_counts.index, band_counts.values,
               color=[band_colors.get(b, "gray") for b in band_counts.index])
        ax.set_ylabel("Hours")
        ax.set_title("Hours per Demand Band")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Forecast Error Distribution")
    if "error_mw" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(df["error_mw"], bins=50, alpha=0.7, color="steelblue")
        ax.axvline(df["error_mw"].mean(), color="red",    linestyle="--", label="Mean error")
        ax.axvline(RMSE_ALERT,           color="orange",  linestyle="--", label=f"Alert ({RMSE_ALERT:.0f} MW)")
        ax.set_xlabel("Forecast Error (MW)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Error Statistics")
    if "error_mw" in df.columns:
        st.write(df["error_mw"].describe().to_frame().T.round(2))
else:
    st.warning("No monitor scores found. Run train_model.py first.")


# ============================================================
# SECTION 4 — PSI DRIFT REPORT
# ============================================================

st.markdown("---")
st.subheader("📉 Feature Drift Report (PSI)")

if os.path.exists(PSI_PATH):
    df_psi = pd.read_csv(PSI_PATH)
    if "drift_score" in df_psi.columns:

        def _psi_flag(val):
            if val >= PSI_HIGH:       return "🔴 CRITICAL"
            elif val >= PSI_MODERATE: return "🟡 MODERATE"
            return "🟢 OK"

        df_psi["status"] = df_psi["drift_score"].apply(_psi_flag)
        st.dataframe(df_psi.head(15), use_container_width=True)

        # ── PSI Bar Chart (UPGRADE — matches Credit Risk dashboard style) ──
        fig, ax = plt.subplots(figsize=(12, 5))
        top     = df_psi.head(15)
        colors  = [
            "#E74C3C" if v >= PSI_HIGH else "#F39C12" if v >= PSI_MODERATE else "#2ECC71"
            for v in top["drift_score"]
        ]
        ax.barh(top["feature"][::-1], top["drift_score"][::-1], color=colors[::-1])
        ax.axvline(PSI_MODERATE, color="orange", linestyle="--", label=f"Moderate ({PSI_MODERATE})")
        ax.axvline(PSI_HIGH,     color="red",    linestyle="--", label=f"Critical ({PSI_HIGH})")
        ax.set_title("Feature PSI Drift Scores (Top 15)")
        ax.set_xlabel("PSI Score")
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
else:
    st.warning("Feature drift report not found. Run train_model.py first.")


# ============================================================
# SECTION 5 — RECENT PREDICTIONS
# ============================================================

st.markdown("---")
st.subheader("📋 Recent API Predictions")

if os.path.exists(LOG_PATH):
    log_df = pd.read_csv(LOG_PATH)
    st.dataframe(log_df.tail(20))
else:
    st.info(
        "Prediction logs are written by the API service. "
        "Run the API and make predictions to see logs here."
    )