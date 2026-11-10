import streamlit as st

st.markdown("# HealthGuard AI")
st.markdown("### AI-Based Disease Risk Prediction and Health Monitoring System")

st.write("""
Welcome to HealthGuard AI, a machine-learning-based health assessment
platform designed to help users explore potential health risks.
""")

st.info("Available assessments: Diabetes, Heart Disease and Hypertension.")

c1, c2, c3 = st.columns(3)
c1.metric("Disease Models", "3")
c2.metric("Platform", "AI / ML")
c3.metric("API", "FastAPI")

st.markdown("### What can you do?")
st.markdown("""
- Create an account and log in.
- Maintain your health profile.
- Explore disease risk predictions.
- View your prediction history.
- Read health recommendations.
- Review machine-learning model performance.
""")

st.warning(
    "Educational demonstration only. Predictions are not medical diagnoses "
    "and must not replace advice from a qualified healthcare professional."
)
