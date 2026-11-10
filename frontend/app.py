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


# Remember the currently selected page
if "selected_nav" not in st.session_state:
    st.session_state["selected_nav"] = list(pages.keys())[0]

with st.sidebar:

    # Branding
    st.markdown(
        """
        <div class="hg-sidebar-brand">
            <div class="hg-sidebar-logo">🏥</div>
            <div>
                <div class="hg-sidebar-title">HealthGuard AI</div>
                <div class="hg-sidebar-subtitle">Disease Risk Prediction</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="hg-sidebar-divider"></div>', unsafe_allow_html=True)

    # User login status
    if st.session_state.get("token"):
        st.success(
            f"Logged in: {st.session_state.get('user_name', 'User')}"
        )

    # Navigation heading
    st.markdown(
        '<div class="hg-sidebar-section">MAIN MENU</div>',
        unsafe_allow_html=True
    )

    # Navigation buttons
    for i, page_label in enumerate(pages.keys()):

        is_active = st.session_state["selected_nav"] == page_label

        with st.container(
            key=f"sidebar_nav_active_{i}" if is_active
            else f"sidebar_nav_item_{i}"
        ):
            if st.button(
                page_label,
                key=f"sidebar_button_{i}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state["selected_nav"] = page_label
                st.rerun()

    # Logout
    if st.session_state.get("token"):
        st.divider()

        if st.button("🚪 Logout", use_container_width=True):
            for key in ["token", "user_name", "user_id", "profile"]:
                st.session_state.pop(key, None)
            st.rerun()

    # Bottom information card
    st.markdown(
        """
        <div class="hg-sidebar-footer">
            <div class="hg-sidebar-footer-title">
                🛡️ HealthGuard AI
            </div>
            <div class="hg-sidebar-footer-text">
                AI-powered health monitoring and disease risk assessment.
            </div>
            <div class="hg-sidebar-footer-version">
                Educational Project · Version 1.0
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

nav = st.session_state["selected_nav"]

# Load and display the selected page
selected_file = os.path.join(PAGES_DIR, pages[nav])

if os.path.isfile(selected_file):
    with open(selected_file, encoding="utf-8") as f:
        page_code = compile(
            f.read(),
            selected_file,
            "exec"
        )

    exec(
        page_code,
        {
            "__name__": "__main__",
            "__file__": selected_file
        }
    )
else:
    st.error(f"Page not found: {selected_file}")

