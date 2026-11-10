import os
import sys
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pages")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="HealthGuard AI",
    page_icon="🏥",
    layout="wide"
)

css_path = os.path.join(os.path.dirname(__file__), "styles", "main.css")
if os.path.exists(css_path):
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

pages = {
    "🏠 Home": "01_home.py",
    "🔐 Login / Register": "02_auth.py",
    "👤 Health Profile": "03_profile.py",
    "🔬 Disease Prediction": "04_prediction.py",
    "📊 Health Dashboard": "05_dashboard.py",
    "💡 Recommendations": "06_recommendations.py",
    "📈 Model Performance": "07_model_performance.py",
}

with st.sidebar:
    st.markdown("# 🏥 HealthGuard AI")
    st.caption("Disease Risk Prediction")
    st.divider()

    if st.session_state.get("token"):
        st.success(f"Logged in: {st.session_state.get('user_name', 'User')}")

    nav = st.radio("Navigate", list(pages.keys()), label_visibility="collapsed")

    if st.session_state.get("token"):
        if st.button("Logout", use_container_width=True):
            for key in ["token", "user_name", "user_id", "profile"]:
                st.session_state.pop(key, None)
            st.rerun()

selected_file = os.path.join(PAGES_DIR, pages[nav])

if os.path.isfile(selected_file):
    with open(selected_file, encoding="utf-8") as f:
        page_code = compile(f.read(), selected_file, "exec")
    exec(
    page_code,
    {
        "__name__": "__main__",
        "__file__": os.path.join(PAGES_DIR, selected_file)
    }
)
else:
    st.error(f"Page file is missing: {selected_file}")
