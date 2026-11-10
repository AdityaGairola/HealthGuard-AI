import streamlit as st
import requests
import os

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api")

st.title("Login / Register")

login_tab, register_tab = st.tabs(["Login", "Create Account"])

with login_tab:
    st.subheader("Welcome Back")

    with st.form("login_form"):
        email = st.text_input("Email Address")
        password = st.text_input("Password", type="password")
        login_submit = st.form_submit_button("Login", use_container_width=True)

    if login_submit:
        if not email or not password:
            st.error("Please enter your email and password.")
        else:
            try:
                response = requests.post(
                    f"{API_BASE}/login",
                    data={"username": email, "password": password},
                    timeout=15
                )
                if response.ok:
                    data = response.json()
                    st.session_state["token"] = data["access_token"]
                    st.session_state["user_name"] = data.get("user_name", "User")
                    st.session_state["user_id"] = data.get("user_id")
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error(response.json().get("detail", "Login failed."))
            except requests.RequestException as e:
                st.error(f"Could not connect to backend: {e}")

with register_tab:
    st.subheader("Create Your Account")

    with st.form("register_form"):
        name = st.text_input("Full Name")
        reg_email = st.text_input("Email Address", key="reg_email")
        reg_password = st.text_input("Password", type="password", key="reg_password")
        confirm_password = st.text_input("Confirm Password", type="password")
        register_submit = st.form_submit_button(
            "Create Account", use_container_width=True
        )

    if register_submit:
        if not name or not reg_email or not reg_password:
            st.error("Please complete all fields.")
        elif len(reg_password) < 6:
            st.error("Password must contain at least 6 characters.")
        elif reg_password != confirm_password:
            st.error("Passwords do not match.")
        else:
            try:
                response = requests.post(
                    f"{API_BASE}/register",
                    json={
                        "name": name,
                        "email": reg_email,
                        "password": reg_password
                    },
                    timeout=15
                )
                if response.status_code == 201:
                    st.success("Account created! Please log in.")
                else:
                    st.error(response.json().get("detail", "Registration failed."))
            except requests.RequestException as e:
                st.error(f"Could not connect to backend: {e}")

if st.session_state.get("token"):
    st.success(f"Logged in as {st.session_state.get('user_name', 'User')}")
