"""
HealthGuard AI — Shared utilities for Streamlit frontend.
"""
import os
import requests
import streamlit as st
from datetime import datetime

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api")


def load_css():
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "styles", "main.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def api_post(endpoint: str, data: dict, token: str = None) -> dict | None:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=data, headers=headers, timeout=15)
        if r.ok:
            return r.json()
        st.error(f"API error {r.status_code}: {r.json().get('detail', r.text)}")
    except requests.ConnectionError:
        st.error("⚠️ Cannot connect to backend. Make sure it's running on http://127.0.0.1:8000")
    return None


def api_get(endpoint: str, token: str = None, params: dict = None) -> dict | list | None:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = requests.get(f"{API_BASE}{endpoint}", headers=headers, params=params, timeout=15)
        if r.ok:
            return r.json()
        st.error(f"API error {r.status_code}: {r.json().get('detail', r.text)}")
    except requests.ConnectionError:
        st.error("⚠️ Cannot connect to backend. Make sure it's running on http://127.0.0.1:8000")
    return None


def api_put(endpoint: str, data: dict, token: str) -> dict | None:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    try:
        r = requests.put(f"{API_BASE}{endpoint}", json=data, headers=headers, timeout=15)
        if r.ok:
            return r.json()
        st.error(f"API error {r.status_code}: {r.json().get('detail', r.text)}")
    except requests.ConnectionError:
        st.error("⚠️ Cannot connect to backend.")
    return None


def login_form():
    """Login form — returns token or None."""
    with st.form("login_form"):
        email    = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log In")
    if submitted:
        result = api_post("/login",
                          {"username": email, "password": password})
        # FastAPI OAuth2 uses form-data, so we call it differently
        if result is None:
            # Retry with form-encoded
            try:
                r = requests.post(
                    f"{API_BASE}/login",
                    data={"username": email, "password": password},
                    timeout=15,
                )
                if r.ok:
                    return r.json()
                st.error(r.json().get("detail", "Login failed"))
            except Exception:
                pass
        return result
    return None


def is_logged_in() -> bool:
    return "token" in st.session_state and st.session_state.token


def require_login():
    if not is_logged_in():
        st.warning("Please log in to access this page.")
        st.stop()


def render_risk_badge(risk_category: str) -> str:
    css_class = {
        "Low":      "risk-low",
        "Moderate": "risk-moderate",
        "High":     "risk-high",
    }.get(risk_category, "risk-moderate")
    return f'<span class="{css_class}">{risk_category} Risk</span>'


def format_date(iso_str: str) -> str:
    try:
        return datetime.fromisoformat(iso_str).strftime("%b %d, %Y %I:%M %p")
    except Exception:
        return iso_str
