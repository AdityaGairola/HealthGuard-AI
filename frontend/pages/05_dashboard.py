
"""Page 5 — Health Monitoring Dashboard"""

import sys
import os
import html

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api")

# ── Configuration ────────────────────────────────────────────────────────────

DISEASE_COLORS = {
    "diabetes": "#3b82f6",
    "heart": "#ef4444",
    "hypertension": "#0d9488",
}

RISK_COLORS = {
    "Low": "#10b981",
    "Moderate": "#f59e0b",
    "High": "#ef4444",
}

PLOT_LAYOUT = {
    "height": 330,
    "margin": dict(l=15, r=15, t=35, b=15),
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "#ffffff",
    "font": dict(
        family="Arial",
        color="#102A43",
        size=15
    ),
    "title_font": dict(color="#102A43", size=18),
    "legend": dict(
        font=dict(color="#102A43", size=14)
    ),
    "hoverlabel": dict(
        bgcolor="#102A43",
        font=dict(color="#ffffff", size=14)
    ),
}

def get_headers():
    return {
        "Authorization": f"Bearer {st.session_state.get('token', '')}"
    }


# ── Page Header ──────────────────────────────────────────────────────────────

st.markdown("""
<div class="hg-header">
    <h1>📊 Health Monitoring Dashboard</h1>
    <p>Your personal overview of health assessments, risk trends and history</p>
</div>
""", unsafe_allow_html=True)


# ── Authentication ───────────────────────────────────────────────────────────

if not st.session_state.get("token"):
    st.warning("Please log in first to view your health dashboard.")
    st.stop()


# ── API Functions ────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def fetch_summary(token):
    try:
        response = requests.get(
            f"{API_BASE}/dashboard/summary",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15
        )
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        return {}


@st.cache_data(ttl=30)
def fetch_history(token):
    try:
        response = requests.get(
            f"{API_BASE}/predictions/history",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, list) else []
    except (requests.RequestException, ValueError):
        return []


# ── Refresh ──────────────────────────────────────────────────────────────────

refresh_col, info_col = st.columns([1, 4])

with refresh_col:
    if st.button("🔄 Refresh Dashboard", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with info_col:
    st.caption("Dashboard data refreshes automatically every 30 seconds.")


# ── Load Data ────────────────────────────────────────────────────────────────

summary = fetch_summary(st.session_state["token"])
history = fetch_history(st.session_state["token"])

if not summary and not history:
    st.warning(
        "Unable to retrieve dashboard data. "
        "Please check that the backend server is running and try refreshing."
    )
    st.stop()


# ── KPI Cards ────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="section-title">📌 Health Overview</div>',
    unsafe_allow_html=True
)

latest = summary.get("latest_predictions", {})

total_assessments = summary.get("total_assessments", 0)

diabetes_risk = latest.get("diabetes", {}).get("risk_category", "—")
heart_risk = latest.get("heart", {}).get("risk_category", "—")

bmi_value = summary.get("bmi")
bmi_display = f"{bmi_value:.1f}" if isinstance(bmi_value, (int, float)) else "—"

c1, c2, c3, c4 = st.columns(4, gap="medium")

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{total_assessments}</div>
        <div class="metric-label">Total Assessments</div>
        <div class="metric-subtitle">All recorded predictions</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="font-size:1.5rem;">
            {html.escape(str(diabetes_risk))}
        </div>
        <div class="metric-label">Latest Diabetes Risk</div>
        <div class="metric-subtitle">Most recent assessment</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="font-size:1.5rem;">
            {html.escape(str(heart_risk))}
        </div>
        <div class="metric-label">Latest Heart Risk</div>
        <div class="metric-subtitle">Most recent assessment</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{bmi_display}</div>
        <div class="metric-label">Your BMI</div>
        <div class="metric-subtitle">Latest available profile value</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("---")


# ── No History State ─────────────────────────────────────────────────────────

if not history:
    st.info(
        "📋 You haven't made any predictions yet. "
        "Visit **Disease Prediction** to complete your first assessment."
    )
    st.stop()


# ── Prepare History Data ──────────────────────────────────────────────────────

df = pd.DataFrame(history)

required_columns = [
    "prediction_date",
    "disease_type",
    "risk_category",
    "model_score"
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(
        "The prediction history is missing required data columns: "
        + ", ".join(missing_columns)
    )
    st.stop()

df["date"] = pd.to_datetime(df["prediction_date"], errors="coerce")
df["model_score"] = pd.to_numeric(df["model_score"], errors="coerce")

df = df.dropna(subset=["date", "model_score"])
df = df.sort_values("date")

if df.empty:
    st.info("There are no valid prediction records to display yet.")
    st.stop()


# ── Charts Row 1 ──────────────────────────────────────────────────────────────

st.markdown(
    '<div class="section-title">📈 Health Analytics</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2, gap="large")


# Risk Score Trends
with col1:
    st.markdown("**Risk Score Trends**")
    st.caption("See how your prediction scores change over time.")

    fig = px.line(
        df,
        x="date",
        y="model_score",
        color="disease_type",
        markers=True,
        labels={
            "model_score": "Risk Score",
            "date": "Assessment Date",
            "disease_type": "Disease"
        },
        color_discrete_map=DISEASE_COLORS
    )

    fig.update_layout(**PLOT_LAYOUT, legend_title="Disease")
    fig.update_yaxes(
    range=[0, 1],
    tickformat=".0%",
    gridcolor="#e2e8f0",
    tickfont=dict(color="#102A43", size=13),
    title_font=dict(color="#102A43", size=14)
)

    fig.update_xaxes(
    gridcolor="#f1f5f9",
    tickfont=dict(color="#102A43", size=13),
    title_font=dict(color="#102A43", size=14)
)

    st.plotly_chart(fig, use_container_width=True)


# Risk Distribution
with col2:
    st.markdown("**Risk Category Distribution**")
    st.caption("Breakdown of your recorded risk categories.")

    risk_counts = df["risk_category"].value_counts()

    risk_colors = [
        RISK_COLORS.get(str(risk), "#94a3b8")
        for risk in risk_counts.index
    ]

    fig2 = go.Figure(
        go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.58,
            marker=dict(
                colors=risk_colors,
                line=dict(color="#ffffff", width=3)
            ),
            textinfo="label+percent",
            textposition="outside"
        )
    )

    fig2.update_layout(
        **PLOT_LAYOUT,
        showlegend=False
    )

    st.plotly_chart(fig2, use_container_width=True)


# ── Charts Row 2 ──────────────────────────────────────────────────────────────

col3, col4 = st.columns(2, gap="large")


# Disease-wise Assessments
with col3:
    st.markdown("**Assessments by Disease**")
    st.caption("Compare the number of assessments for each disease.")

    disease_counts = df["disease_type"].value_counts().reset_index()
    disease_counts.columns = ["Disease", "Count"]

    fig3 = px.bar(
        disease_counts,
        x="Disease",
        y="Count",
        color="Disease",
        color_discrete_map=DISEASE_COLORS,
        text="Count"
    )

    fig3.update_traces(
        textposition="outside",
        marker_line_width=0
    )

    fig3.update_layout(
        **PLOT_LAYOUT,
        showlegend=False
    )

    fig3.update_yaxes(
    rangemode="tozero",
    gridcolor="#e2e8f0",
    tickfont=dict(color="#102A43", size=13),
    title_font=dict(color="#102A43", size=14)
)

    fig3.update_xaxes(
    showgrid=False,
    tickfont=dict(color="#102A43", size=13),
    title_font=dict(color="#102A43", size=14)
)

    st.plotly_chart(fig3, use_container_width=True)


# Recent Assessments
with col4:
    st.markdown("**Recent Assessments**")
    st.caption("Your 10 most recent prediction records.")

    display_df = df[
        ["date", "disease_type", "risk_category", "model_score"]
    ].copy()

    display_df = display_df.sort_values("date", ascending=False).head(10)

    display_df["date"] = display_df["date"].dt.strftime("%b %d, %Y")
    display_df["model_score"] = display_df["model_score"].map(
        lambda value: f"{value:.1%}"
    )

    display_df.columns = ["Date", "Disease", "Risk Category", "Score"]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=350
    )


# ── Date Range Filter ────────────────────────────────────────────────────────

st.markdown("---")

st.markdown(
    '<div class="section-title">🗓️ Explore Assessment History</div>',
    unsafe_allow_html=True
)

st.caption("Choose a date range to examine your prediction scores.")

min_date = df["date"].min().date()
max_date = df["date"].max().date()

date_range = st.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range

    filtered = df[
        (df["date"].dt.date >= start_date) &
        (df["date"].dt.date <= end_date)
    ]

    st.write(
        f"Showing **{len(filtered)}** assessment(s) "
        f"from **{start_date.strftime('%b %d, %Y')}** "
        f"to **{end_date.strftime('%b %d, %Y')}**."
    )

    if not filtered.empty:
        fig4 = px.scatter(
            filtered,
            x="date",
            y="model_score",
            color="disease_type",
            symbol="risk_category",
            hover_data=["risk_category"],
            labels={
                "date": "Assessment Date",
                "model_score": "Risk Score",
                "disease_type": "Disease",
                "risk_category": "Risk Category"
            },
            color_discrete_map=DISEASE_COLORS
        )

        fig4.update_layout(**PLOT_LAYOUT)
        fig4.update_yaxes(
    range=[0, 1],
    tickformat=".0%",
    gridcolor="#e2e8f0",
    tickfont=dict(color="#102A43", size=13),
    title_font=dict(color="#102A43", size=14)
)

        fig4.update_xaxes(
    gridcolor="#f1f5f9",
    tickfont=dict(color="#102A43", size=13),
    title_font=dict(color="#102A43", size=14)
)

        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No assessments were found in this date range.")
