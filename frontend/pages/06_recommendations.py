
"""Page 6 — Personalized Health Recommendations"""

import sys
import os
import html

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import requests

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api")

RISK_COLORS = {
    "Low": "#10b981",
    "Moderate": "#f59e0b",
    "High": "#ef4444"
}


def get_headers():
    return {
        "Authorization": f"Bearer {st.session_state.get('token', '')}"
    }


# ── Page Header ──────────────────────────────────────────────────────────────

st.markdown("""
<div class="hg-header">
    <h1>💡 Personalized Health Recommendations</h1>
    <p>Evidence-based lifestyle suggestions tailored to your risk profile</p>
</div>
""", unsafe_allow_html=True)


# ── Authentication ───────────────────────────────────────────────────────────

if not st.session_state.get("token"):
    st.warning("Please log in first to view your recommendations.")
    st.stop()


# ── Fetch Prediction History ─────────────────────────────────────────────────

history = []

try:
    response = requests.get(
        f"{API_BASE}/predictions/history",
        headers=get_headers(),
        timeout=15
    )

    response.raise_for_status()
    data = response.json()

    if isinstance(data, list):
        history = data

except requests.RequestException:
    st.error(
        "Unable to retrieve your assessment history. "
        "Please check that the backend server is running."
    )
    st.stop()

except ValueError:
    st.error("The server returned an invalid response.")
    st.stop()


# ── General Health Guidelines ────────────────────────────────────────────────

GENERAL = [
    (
        "🏃 Regular Exercise",
        "Aim for at least 150 minutes of moderate-intensity aerobic activity "
        "per week. Include muscle-strengthening activities 2+ days per week."
    ),
    (
        "🥗 Balanced Diet",
        "Follow a diet rich in fruits, vegetables, whole grains, and lean "
        "proteins. Limit saturated fats, sodium, and added sugars."
    ),
    (
        "💧 Stay Hydrated",
        "Drink adequate water daily. Avoid sugary beverages and excessive "
        "caffeine."
    ),
    (
        "😴 Quality Sleep",
        "Aim for 7–9 hours of quality sleep per night. Poor sleep is linked "
        "to increased risk of diabetes and cardiovascular disease."
    ),
    (
        "🧘 Stress Management",
        "Practice relaxation techniques such as meditation, deep breathing, "
        "or yoga."
    ),
    (
        "🚭 Avoid Smoking",
        "Smoking significantly increases cardiovascular disease and "
        "hypertension risk. Seek cessation support if you smoke."
    ),
    (
        "🩺 Regular Checkups",
        "Schedule regular health screenings including blood pressure, blood "
        "glucose, and cholesterol tests with your healthcare provider."
    ),
    (
        "⚖️ Healthy Weight",
        "Maintaining a healthy weight can help reduce risk across the "
        "conditions modelled in this application."
    ),
]

st.markdown(
    '<div class="section-title">🌟 General Health Guidelines</div>',
    unsafe_allow_html=True
)

st.caption(
    "Simple everyday habits that support your overall health and wellbeing."
)

general_col1, general_col2 = st.columns(2, gap="large")

for index, (title, description) in enumerate(GENERAL):
    target_col = general_col1 if index % 2 == 0 else general_col2

    with target_col:
        with st.expander(title, expanded=False):
            st.write(description)


st.markdown("---")


# ── No Prediction History ────────────────────────────────────────────────────

if not history:
    st.info(
        "📋 You haven't completed any disease assessments yet. "
        "Visit **Disease Prediction** to receive recommendations based "
        "on your assessment results."
    )

else:

    # ── Recommendation Data ──────────────────────────────────────────────────

    RECS = {
        "diabetes": {
            "Low": [
                "Maintain current healthy habits — your diabetes risk is low.",
                "Continue monitoring glucose if you have a family history of diabetes.",
                "Annual fasting glucose screening is recommended."
            ],
            "Moderate": [
                "Reduce refined sugars and processed carbohydrates.",
                "Increase daily physical activity to 30+ minutes.",
                "Get a fasting blood glucose test with your doctor.",
                "Consider a referral to a registered dietitian."
            ],
            "High": [
                "Consult a healthcare professional about your elevated diabetes risk.",
                "Discuss a structured diet and exercise plan with your healthcare provider.",
                "Ask your doctor about HbA1c and fasting glucose tests.",
                "Discuss appropriate treatment options with your healthcare professional."
            ],
        },

        "heart": {
            "Low": [
                "Keep up heart-healthy habits — your cardiovascular risk is currently low.",
                "Annual blood pressure and cholesterol checks are recommended.",
                "Maintain a Mediterranean or DASH-style diet."
            ],
            "Moderate": [
                "Get cholesterol and blood pressure checked.",
                "Reduce saturated fats and sodium in your diet.",
                "Engage in 150 minutes of aerobic exercise weekly.",
                "Discuss your risk with your doctor."
            ],
            "High": [
                "Arrange a cardiovascular consultation with a healthcare professional.",
                "Ask your doctor whether a lipid panel or ECG is appropriate.",
                "Discuss a heart-healthy diet and lifestyle plan.",
                "Seek medical guidance before beginning strenuous exercise."
            ],
        },

        "hypertension": {
            "Low": [
                "Your hypertension risk is currently low — keep up healthy lifestyle habits.",
                "Monitor blood pressure periodically.",
                "Maintain sodium intake below 2300 mg per day."
            ],
            "Moderate": [
                "Monitor blood pressure at home 2–3 times per week.",
                "Adopt the DASH diet (Dietary Approaches to Stop Hypertension).",
                "Limit excessive alcohol and caffeine intake.",
                "Seek medical advice if readings are consistently elevated."
            ],
            "High": [
                "Consult a healthcare professional about your elevated hypertension risk.",
                "Monitor blood pressure and record your readings.",
                "Discuss a DASH-style diet and sodium reduction with your healthcare provider.",
                "Discuss further evaluation and treatment options with your doctor."
            ],
        },
    }

    disease_icons = {
        "diabetes": "🩸",
        "heart": "❤️",
        "hypertension": "🫀"
    }

    disease_titles = {
        "diabetes": "Diabetes",
        "heart": "Heart Disease",
        "hypertension": "Hypertension"
    }

    # ── Find Latest Assessment for Each Disease ──────────────────────────────

    def assessment_date(record):
        return (
            record.get("prediction_date")
            or record.get("date")
            or ""
        )

    sorted_history = sorted(
        history,
        key=lambda record: str(assessment_date(record)),
        reverse=True
    )

    latest_by_disease = {}

    for record in sorted_history:
        disease = record.get("disease_type", "").lower()

        if disease and disease not in latest_by_disease:
            latest_by_disease[disease] = record

    # ── Personalized Recommendations ────────────────────────────────────────

    st.markdown(
        '<div class="section-title">🎯 Your Personalized Recommendations</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Based on the latest available prediction for each disease."
    )

    if not latest_by_disease:
        st.info("No valid disease assessment records were found.")

    else:

        for disease, record in latest_by_disease.items():

            risk = record.get("risk_category", "Moderate")
            risk = str(risk).title()

            if risk not in RISK_COLORS:
                risk = "Moderate"

            color = RISK_COLORS[risk]

            icon = disease_icons.get(disease, "🏥")
            title = disease_titles.get(disease, disease.title())

            score = record.get("model_score", 0)

            try:
                score = float(score)
                score_display = f"{score:.1%}"
            except (ValueError, TypeError):
                score_display = "—"

            date = str(assessment_date(record))[:10]
            if not date:
                date = "Not available"

            safe_title = html.escape(title)
            safe_risk = html.escape(risk)
            safe_date = html.escape(date)

            st.markdown(
                f"**{icon} {safe_title}**  \n"
                f"Latest assessment: {safe_date}  \n"
                f"**{safe_risk} Risk** | Model Score: {score_display}"
            )

            specific = RECS.get(disease, {}).get(risk, [])

            if specific:
                st.markdown(
                    '<div class="recommendation-list-title">'
                    'Recommended Actions'
                    '</div>',
                    unsafe_allow_html=True
                )

                
                for number, suggestion in enumerate(specific, start=1):
                    st.markdown(
                        f"""
                        <div class="recommendation-item">
                            <span class="recommendation-number">{number}</span>
                            <span>{html.escape(str(suggestion))}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


            else:
                st.info(
                    f"No specific recommendations are available for {title}."
                )

            st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)


# ── Medical Disclaimer ───────────────────────────────────────────────────────

st.markdown("---")

st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Important Disclaimer:</strong>
    All recommendations above are general health information and are
    not personalized medical advice. This application does not diagnose
    diseases or prescribe medications or treatments. Always consult a
    qualified healthcare professional before making health decisions.
</div>
""", unsafe_allow_html=True)
