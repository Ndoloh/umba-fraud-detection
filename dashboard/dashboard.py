import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🚨",
    layout="wide",
)

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "predictions.csv")

# Fallback: allow running standalone from the dashboard/ folder during dev/testing
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(BASE_DIR, "predictions.csv")


@st.cache_data
def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


try:
    df = load_predictions(DATA_PATH)
except FileNotFoundError:
    st.error(f"Could not find predictions.csv at: {DATA_PATH}")
    st.stop()

DEFAULT_THRESHOLD = 0.7201

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("🚨 Fraud Detection Dashboard")
st.caption(
    "Operational view of model-scored transactions. "
    "Use the threshold slider in the sidebar to see how the alarm rate changes."
)

# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------
st.sidebar.header("Settings")

threshold = st.sidebar.slider(
    "Fraud alarm threshold",
    min_value=0.0,
    max_value=1.0,
    value=float(DEFAULT_THRESHOLD),
    step=0.01,
    help="Transactions with a fraud probability at or above this value are flagged as alarms.",
)

st.sidebar.markdown(f"**Selected threshold:** `{threshold:.4f}`")
st.sidebar.markdown(f"**Model-selected threshold:** `{DEFAULT_THRESHOLD}`")

top_n = st.sidebar.number_input(
    "Number of high-risk transactions to display",
    min_value=5,
    max_value=200,
    value=25,
    step=5,
)

# ---------------------------------------------------------
# Derived fields
# ---------------------------------------------------------
df["alarm"] = df["isFraud_prob"] >= threshold

total_txns = len(df)
flagged_txns = int(df["alarm"].sum())
flagged_rate = flagged_txns / total_txns if total_txns else 0.0
avg_prob = df["isFraud_prob"].mean()
median_prob = df["isFraud_prob"].median()

# ---------------------------------------------------------
# KPI row
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions Scored", f"{total_txns:,}")
col2.metric("Flagged as Fraud", f"{flagged_txns:,}")
col3.metric("Predicted Fraud Rate", f"{flagged_rate:.2%}")
col4.metric("Average Fraud Probability", f"{avg_prob:.3f}")

st.divider()

# ---------------------------------------------------------
# Probability distribution
# ---------------------------------------------------------
st.subheader("Fraud Probability Distribution")

fig_hist = px.histogram(
    df,
    x="isFraud_prob",
    nbins=50,
    labels={"isFraud_prob": "Predicted Fraud Probability"},
    title="Distribution of Predicted Fraud Probabilities",
)
fig_hist.add_vline(
    x=threshold,
    line_width=2,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Threshold ({threshold:.4f})",
    annotation_position="top right",
)
fig_hist.update_layout(
    bargap=0.02,
    xaxis_title="Predicted Fraud Probability",
    yaxis_title="Number of Transactions",
)
st.plotly_chart(fig_hist, use_container_width=True)

with st.expander("Distribution summary statistics"):
    st.write(df["isFraud_prob"].describe().to_frame(name="isFraud_prob"))

st.divider()

# ---------------------------------------------------------
# Flagged vs not-flagged breakdown
# ---------------------------------------------------------
st.subheader("Alarm Breakdown")

breakdown_col1, breakdown_col2 = st.columns([1, 2])

with breakdown_col1:
    pie_df = pd.DataFrame({
        "Status": ["Flagged (Alarm)", "Not Flagged"],
        "Count": [flagged_txns, total_txns - flagged_txns],
    })
    fig_pie = px.pie(
        pie_df,
        names="Status",
        values="Count",
        color="Status",
        color_discrete_map={"Flagged (Alarm)": "#d62728", "Not Flagged": "#2ca02c"},
        title="Share of Transactions Flagged",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with breakdown_col2:
    st.markdown("**What this means for operations:**")
    st.markdown(
        f"""
        - At a threshold of **{threshold:.4f}**, **{flagged_txns:,}** out of
          **{total_txns:,}** transactions ({flagged_rate:.2%}) are flagged for review.
        - Lowering the threshold will flag more transactions (higher recall of
          fraud, but more false alarms to review).
        - Raising the threshold will flag fewer transactions (fewer alarms to
          review, but higher risk of missing actual fraud).
        - The median predicted probability across all transactions is
          **{median_prob:.3f}**.
        """
    )

st.divider()

# ---------------------------------------------------------
# Top high-risk transactions
# ---------------------------------------------------------
st.subheader(f"Top {top_n} High-Risk Transactions")

top_risky = (
    df.sort_values("isFraud_prob", ascending=False)
    .head(top_n)
    .reset_index(drop=True)
)
top_risky.index = top_risky.index + 1  # 1-indexed for readability
top_risky_display = top_risky.copy()
top_risky_display["isFraud_prob"] = top_risky_display["isFraud_prob"].round(4)
top_risky_display["alarm"] = top_risky_display["alarm"].map({True: "🚨 Yes", False: "No"})

st.dataframe(
    top_risky_display.rename(columns={
        "TransactionID": "Transaction ID",
        "isFraud_prob": "Fraud Probability",
        "alarm": "Flagged",
    }),
    use_container_width=True,
)

csv_export = top_risky.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download high-risk transactions as CSV",
    data=csv_export,
    file_name="top_high_risk_transactions.csv",
    mime="text/csv",
)

st.divider()

# ---------------------------------------------------------
# Note on scope / data availability
# ---------------------------------------------------------
st.info(
    "This dashboard reflects model-predicted fraud probabilities only. "
    "The current predictions file does not include true fraud labels or "
    "additional transaction context (amount, country, channel, etc.), so "
    "accuracy metrics (precision, recall, confusion matrix) and enriched "
    "transaction details are not shown. If a labeled dataset or richer "
    "transaction context becomes available, this dashboard can be extended "
    "to include model performance evaluation and deeper transaction drill-down."
)
