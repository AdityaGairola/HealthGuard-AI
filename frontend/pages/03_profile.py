"""Page 3 — Health Profile"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import requests
import json

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api")

def get_headers():
    return {"Authorization": f"Bearer {st.session_state.get('token','')}"}

st.markdown("""
<div class="hg-header">
    <h1>👤 Health Profile</h1>
    <p>Manage your personal health information</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.get("token"):
    st.warning("⚠️ Please log in first.")
    st.stop()

# Load existing profile
existing = {}
try:
    r = requests.get(f"{API_BASE}/profile", headers=get_headers(), timeout=10)
    if r.ok:
        existing = r.json()
except:
    pass

with st.form("profile_form"):
    st.markdown("### 📋 Personal Information")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 1, 120, value=int(existing.get("age") or 25))
    with c2:
        gender = st.selectbox("Gender", ["Male","Female","Other","Prefer not to say"],
                              index=["Male","Female","Other","Prefer not to say"].index(
                                  existing.get("gender","Male")) if existing.get("gender") in
                                  ["Male","Female","Other","Prefer not to say"] else 0)
    with c3:
        pass

    st.markdown("### 📏 Body Measurements")
    c1, c2 = st.columns(2)
    with c1:
        height = st.number_input("Height (cm)", 50.0, 300.0,
                                  value=float(existing.get("height_cm") or 170.0), step=0.5)
    with c2:
        weight = st.number_input("Weight (kg)", 10.0, 500.0,
                                  value=float(existing.get("weight_kg") or 70.0), step=0.5)

    if height > 0:
        bmi_calc = round(weight / ((height/100)**2), 2)
        bmi_cat = ("Underweight" if bmi_calc < 18.5 else
                   "Normal" if bmi_calc < 25 else
                   "Overweight" if bmi_calc < 30 else "Obese")
        st.info(f"📊 Calculated BMI: **{bmi_calc}** ({bmi_cat})")

    st.markdown("### 🏃 Lifestyle")
    c1, c2 = st.columns(2)
    with c1:
        smoking = st.checkbox("Current Smoker", value=bool(existing.get("smoking", False)))
    with c2:
        activity = st.selectbox("Physical Activity Level",
                                 ["low","moderate","high"],
                                 index=["low","moderate","high"].index(
                                     existing.get("physical_activity","moderate")))

    st.markdown("### 🧬 Medical History")
    fh_diabetes = st.checkbox("Family history of Diabetes",
                               value=json.loads(existing.get("family_history") or "{}").get("diabetes", False))
    fh_heart    = st.checkbox("Family history of Heart Disease",
                               value=json.loads(existing.get("family_history") or "{}").get("heart", False))
    fh_htn      = st.checkbox("Family history of Hypertension",
                               value=json.loads(existing.get("family_history") or "{}").get("hypertension", False))
    medical_history = st.text_area("Other medical notes (optional)",
                                    value=existing.get("medical_history",""), height=80)

    submitted = st.form_submit_button("💾 Save Profile", use_container_width=True)

if submitted:
    payload = {
        "age": age, "gender": gender,
        "height_cm": height, "weight_kg": weight,
        "smoking": smoking, "physical_activity": activity,
        "family_history": {"diabetes": fh_diabetes, "heart": fh_heart, "hypertension": fh_htn},
        "medical_history": medical_history,
    }
    try:
        r = requests.put(f"{API_BASE}/profile", json=payload, headers=get_headers(), timeout=10)
        if r.ok:
            st.success("✅ Profile saved successfully!")
            st.session_state["profile"] = r.json()
        else:
            try:
               error_data = r.json()
               st.error(error_data.get("detail", "Save failed."))
            except ValueError:
               st.error(f"Server returned HTTP {r.status_code}:")
               st.code(r.text or "The server returned an empty response.")
    except requests.ConnectionError:
        st.error("⚠️ Cannot reach backend.")

st.markdown("---")
st.markdown("""
<div class="disclaimer">
    Your data is stored locally in the SQLite database and is only accessible to your account.
</div>
""", unsafe_allow_html=True)
