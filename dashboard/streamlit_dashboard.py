import sys
import os
import streamlit as st
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from storage.session_history import SessionHistory
from analytics.workout_analytics import WorkoutStatistics, WorkoutSession
from analytics.performance_analytics import PerformanceAnalytics
from analytics.progress_prediction import ProgressPredictor
from analytics.report_generator import ReportGenerator

st.set_page_config(
    page_title="Workout Analytics",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dark theme CSS ──────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f0f0f; }
    .metric-card {
        background: #1a1a1a;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stMetric { background: #1a1a1a; border-radius: 8px; padding: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Load Data ───────────────────────────────────────────
@st.cache_data(ttl=30)
def load_sessions():
    history = SessionHistory()
    return history.all()

sessions = load_sessions()

# ── Sidebar ─────────────────────────────────────────────
st.sidebar.title("🏋️ Workout Analytics")
st.sidebar.markdown("---")

if sessions:
    exercises = sorted(set(s.exercise for s in sessions))
    selected_exercise = st.sidebar.selectbox("Select Exercise", exercises)
    date_filter = st.sidebar.selectbox(
        "Time Range",
        ["All Time", "Last 7 Days", "Last 30 Days", "Last 90 Days"]
    )
    from datetime import timedelta
    now = datetime.now()
    cutoff_map = {
        "All Time": datetime(2000, 1, 1),
        "Last 7 Days": now - timedelta(days=7),
        "Last 30 Days": now - timedelta(days=30),
        "Last 90 Days": now - timedelta(days=90),
    }
    cutoff = cutoff_map[date_filter]
    filtered = [s for s in sessions if s.date >= cutoff]
else:
    filtered = []
    selected_exercise = None

# ── Main Content ─────────────────────────────────────────
st.title("🏋️ Workout Analytics Dashboard")

if not sessions:
    st.warning("No sessions found. Add workouts via the API or CSV.")
    st.stop()

stats = WorkoutStatistics(filtered)
summary = stats.summary()

# ── Top Metrics ──────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Sessions", summary["total_sessions"])
col2.metric("Total Duration", f"{summary['total_duration_min']} min")
col3.metric("Total Calories", f"{summary['total_calories']} kcal")

perf = PerformanceAnalytics(filtered)
col4.metric("Consistency Score", f"{perf.consistency_score()} / 100")

st.markdown("---")

# ── Charts ───────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader(f"💪 Strength Progression — {selected_exercise}")
    ex_sessions = [s for s in filtered if s.exercise == selected_exercise]
    if ex_sessions:
        df = pd.DataFrame({
            "Date": [s.date for s in ex_sessions],
            "Max Weight (kg)": [stats.max_weight(s) for s in ex_sessions],
            "Volume": [stats.total_volume(s) for s in ex_sessions],
        })
        fig = px.line(df, x="Date", y="Max Weight (kg)",
                      markers=True, template="plotly_dark",
                      color_discrete_sequence=["#00e5ff"])
        fig.update_layout(paper_bgcolor="#1a1a1a", plot_bgcolor="#1a1a1a")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data for selected exercise.")

with col_right:
    st.subheader("📊 Weekly Volume")
    if selected_exercise:
        weekly = stats.weekly_volume(selected_exercise)
        if weekly:
            df_weekly = pd.DataFrame(list(weekly.items()), columns=["Week", "Volume"])
            fig2 = px.bar(df_weekly, x="Week", y="Volume",
                          template="plotly_dark",
                          color_discrete_sequence=["#ff6b35"])
            fig2.update_layout(paper_bgcolor="#1a1a1a", plot_bgcolor="#1a1a1a")
            st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── Prediction ───────────────────────────────────────────
st.subheader("🔮 Progress Prediction")
predictor = ProgressPredictor(filtered)
pred = predictor.predict_max_weight(selected_exercise, future_sessions=10)

if "error" not in pred:
    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("Current Max", f"{pred['current_max']} kg")
    pc2.metric("Predicted Max (10 sessions)", f"{pred['predicted_max']} kg")
    pc3.metric("Estimated Gain", f"+{pred['gain_estimate']} kg")

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        y=[stats.max_weight(s) for s in [s for s in filtered if s.exercise == selected_exercise]],
        name="Actual", line=dict(color="#00e5ff", width=2)
    ))
    fig3.add_trace(go.Scatter(
        y=pred["predicted_values"], name="Predicted",
        line=dict(color="#ff6b35", width=2, dash="dash")
    ))
    fig3.update_layout(template="plotly_dark", paper_bgcolor="#1a1a1a",
                       plot_bgcolor="#1a1a1a", title="Actual vs Predicted")
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ── Full Report ──────────────────────────────────────────
st.subheader("📋 Full Report")
if st.button("Generate Text Report"):
    rg = ReportGenerator(filtered)
    st.code(rg.to_text(), language="text")