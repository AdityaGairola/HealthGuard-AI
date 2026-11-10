import plotly.express as px
import streamlit as st
import requests
import os
import pandas as pd
import numpy as np

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api")

# ---------------------------------------------------------
# PAGE HEADER
# ---------------------------------------------------------

st.markdown("""
<div class="hg-header">
    <h1>📊 Model Performance</h1>
    <p>Evaluate and compare the performance of our trained AI models.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("")

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def format_metric(value):
    """Convert model metrics into readable percentages."""
    if value is None:
        return "N/A"

    try:
        value = float(value)

        if value < 0:
            return "N/A"

        if value <= 1:
            value *= 100

        return f"{value:.2f}%"

    except (ValueError, TypeError):
        return "N/A"


def metric_number(value):
    """Convert a metric into a numeric percentage for charts."""
    try:
        value = float(value)

        if value < 0:
            return None

        return value * 100 if value <= 1 else value

    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------
# FETCH MODEL PERFORMANCE DATA
# ---------------------------------------------------------

try:
    response = requests.get(
        f"{API_BASE}/model/performance",
        timeout=15
    )

    if not response.ok:
        st.error(
            f"API returned status {response.status_code}: "
            f"{response.text[:300]}"
        )

    else:
        results = response.json()

        if not isinstance(results, dict) or not results:
            st.info("No model performance information is available yet.")

        else:

            for disease, details in results.items():

                if not isinstance(details, dict):
                    continue

                disease_name = disease.replace("_", " ").title()

                st.markdown("---")

                st.subheader(f"🧬 {disease_name}")

                if "error" in details:
                    st.warning(str(details["error"]))
                    continue

                # -------------------------------------------------
                # BEST MODEL INFORMATION
                # -------------------------------------------------

                metrics = details.get("best_metrics", {})

                if not isinstance(metrics, dict):
                    metrics = {}

                best_model = details.get(
                    "best_model",
                    metrics.get("model", "Not available")
                )

                st.markdown("#### Best Performing Model")

                with st.container(border=True):
                    st.markdown(f"**Selected Model:** {best_model}")

                    st.caption(
                        "Performance metrics calculated from the model's test data."
                    )
                    with st.container(key=f"model_performance_metrics_{disease}"):
                        cols = st.columns(4)

                        metric_items = [
                            ("accuracy", "Accuracy"),
                            ("precision", "Precision"),
                            ("recall", "Recall"),
                            ("f1", "F1 Score")
                        ]

                        for col, (key, label) in zip(cols, metric_items):

                            value = metrics.get(key)

                            with col:
                                with st.container(border=True):
                                    st.metric(
                                        label=label,
                                        value=format_metric(value)
                                    )

                # -------------------------------------------------
                # MODEL COMPARISON
                # -------------------------------------------------

                all_results = details.get("all_results", [])

                if isinstance(all_results, list) and all_results:

                    st.markdown("#### Model Comparison")

                    df = pd.DataFrame(all_results)
                    # Add cross-validation F1 scores when available.
                    cv_scores = details.get("cross_validation_f1", {})

                    if isinstance(cv_scores, dict) and cv_scores:
                        df["cross_validation_f1"] = df["model"].map(cv_scores)

                    display_cols = [
                        c for c in [
                            "model",
                            "accuracy",
                            "precision",
                            "recall",
                            "f1",
                            "cross_validation_f1",
                            "roc_auc"
                        ]
                        if c in df.columns
                    ]

                    if display_cols:

                        display_df = df[display_cols].copy()

                        # Format metric columns as percentages.
                        for column in [
                            "accuracy",
                            "precision",
                            "recall",
                            "f1",
                            "roc_auc"
                        ]:
                            if column in display_df.columns:
                                display_df[column] = display_df[column].apply(
                                    format_metric
                                )
                        if "cross_validation_f1" in display_df.columns:
                            display_df["cross_validation_f1"] = display_df[
                                "cross_validation_f1"
                            ].apply(
                                lambda value: format_metric(value)
                                if pd.notna(value) else "—"
                            )
                        display_df = display_df.rename(
                            columns={
                                "model": "Model",
                                "accuracy": "Accuracy",
                                "precision": "Precision",
                                "recall": "Recall",
                                "f1": "F1 Score",
                                "cross_validation_f1": "CV F1 Score",
                                "roc_auc": "ROC AUC"
                            }
                        )

                        with st.container(key=f"model_performance_table_{disease}"):
                            st.dataframe(
                                display_df,
                                use_container_width=True,
                                hide_index=True
                            )

                        # -------------------------------------------------
                        # ACCURACY COMPARISON CHART
                        # -------------------------------------------------

                        # -------------------------------------------------
                        # CONFUSION MATRIX
                        # -------------------------------------------------

                        best_metrics = details.get("best_metrics", {})
                        confusion_matrix = best_metrics.get("confusion_matrix")

                        if confusion_matrix:
                            st.markdown("#### Confusion Matrix")
                            st.caption(
                                f"Selected model: {best_model}"
                            )

                            cm = np.array(confusion_matrix)

                            fig_cm = px.imshow(
                                cm,
                                text_auto=True,
                                x=["Predicted Negative", "Predicted Positive"],
                                y=["Actual Negative", "Actual Positive"],
                                color_continuous_scale=["#EAF4F4", "#087E8B"],
                                labels=dict(
                                    x="Predicted Class",
                                    y="Actual Class",
                                    color="Count"
                                ),
                                aspect="auto"
                            )

                            
                            fig_cm.update_layout(
                                height=350,
                                plot_bgcolor="white",
                                paper_bgcolor="white",
                                font=dict(
                                    family="Arial",
                                    color="#102A43",
                                    size=14
                                ),
                                margin=dict(l=30, r=30, t=35, b=30),
                                xaxis=dict(
                                    tickfont=dict(
                                        color="#102A43",
                                        size=13
                                    ),
                                    title_font=dict(
                                        color="#102A43",
                                        size=14
                                    )
                                ),
                                yaxis=dict(
                                    tickfont=dict(
                                        color="#102A43",
                                        size=13
                                    ),
                                    title_font=dict(
                                        color="#102A43",
                                        size=14
                                    )
                                ),
                                coloraxis_colorbar=dict(
                                    tickfont=dict(color="#102A43"),
                                    title_font=dict(color="#102A43")
                                )
                            )

                            fig_cm.update_traces(
                                textfont=dict(
                                    color="#102A43",
                                    size=18
                                )
                            )


                            st.plotly_chart(
                                fig_cm,
                                use_container_width=True,
                                key=f"confusion_matrix_{disease}"
                            )


                        if "accuracy" in df.columns and "model" in df.columns:

                            chart_df = df[["model", "accuracy"]].copy()

                            chart_df["accuracy"] = chart_df["accuracy"].apply(
                                metric_number
                            )

                            chart_df = chart_df.dropna()

                            if not chart_df.empty:

                                st.markdown("#### Accuracy Comparison")

                                chart_df = chart_df.set_index("model")

                                chart_df = chart_df.reset_index()

                            fig = px.bar(
                                chart_df,
                                x="model",
                                y="accuracy",
                                text="accuracy",
                                labels={
                                    "model": "Machine Learning Model",
                                    "accuracy": "Accuracy (%)"
                                }
                            )

                            fig.update_traces(
                                width=0.45,
                                texttemplate="%{text:.1f}%",
                                textfont=dict(color="#102A43", size=13),
                                textposition="outside"
                                
                            )

                            fig.update_layout(
                            height=380,
                            showlegend=False,
                            plot_bgcolor="white",
                            paper_bgcolor="white",

                            font=dict(
                                family="Arial",
                                color="#102A43",
                                size=14
                            ),

                            margin=dict(l=25, r=25, t=45, b=55),

                            xaxis=dict(
                                title=dict(
                                    text="Machine Learning Model",
                                    font=dict(color="#102A43", size=14)
                                ),
                                tickfont=dict(
                                    color="#102A43",
                                    size=13
                                )
                            ),

                            yaxis=dict(
                                title=dict(
                                    text="Accuracy (%)",
                                    font=dict(color="#102A43", size=14)
                                ),
                                tickfont=dict(
                                    color="#102A43",
                                    size=13
                                ),
                                range=[0, 110],
                                gridcolor="#E2E8F0"
                            ),

                            hoverlabel=dict(
                                bgcolor="#102A43",
                                font=dict(color="white", size=13)
                            )
                        )

                            with st.container(key=f"model_performance_chart_{disease}"):
                                st.plotly_chart(fig, use_container_width=True)

                else:
                    st.info(
                        "Detailed model comparison data is not available."
                    )
                
                # -------------------------------------------------
                # FEATURE IMPORTANCE
                # -------------------------------------------------

                feature_importance = details.get("feature_importance", {})

                if isinstance(feature_importance, dict) and feature_importance:

                    st.markdown("#### Feature Importance")

                    importance_df = pd.DataFrame(
    list(feature_importance.items()),
    columns=["Feature", "Importance"]
)

# Convert technical one-hot encoded names into readable labels
                    feature_labels = {
                        "num__age": "Age",
                        "num__trestbps": "Resting Blood Pressure",
                        "num__chol": "Cholesterol",
                        "num__thalach": "Maximum Heart Rate",
                        "num__oldpeak": "ST Depression",
                        "num__ca": "Number of Major Vessels",

                        "cat__sex_0": "Sex: Female",
                        "cat__sex_1": "Sex: Male",
                        "cat__cp_0": "Chest Pain: Typical Angina",
                        "cat__cp_1": "Chest Pain: Atypical Angina",
                        "cat__cp_2": "Chest Pain: Non-Anginal",
                        "cat__cp_3": "Chest Pain: Asymptomatic",
                        "cat__fbs_0": "Fasting Blood Sugar: Normal",
                        "cat__fbs_1": "Fasting Blood Sugar: High",
                        "cat__restecg_0": "ECG: Normal",
                        "cat__restecg_1": "ECG: ST-T Abnormality",
                        "cat__restecg_2": "ECG: LV Hypertrophy",
                        "cat__exang_0": "Exercise Angina: No",
                        "cat__exang_1": "Exercise Angina: Yes",
                        "cat__slope_0": "ST Slope: Upsloping",
                        "cat__slope_1": "ST Slope: Flat",
                        "cat__slope_2": "ST Slope: Downsloping",
                        "cat__thal_3": "Thalassemia: Normal",
                        "cat__thal_6": "Thalassemia: Fixed Defect",
                        "cat__thal_7": "Thalassemia: Reversible Defect",
                    }

                    importance_df["Feature"] = importance_df["Feature"].replace(feature_labels)
                                        

                    importance_df["Importance"] = pd.to_numeric(
                        importance_df["Importance"],
                        errors="coerce"
                    )

                    importance_df = importance_df.dropna()

                    importance_df = importance_df.sort_values(
                        by="Importance",
                        ascending=True
                    )

                    if not importance_df.empty:

                        fig_importance = px.bar(
                            importance_df,
                            x="Importance",
                            y="Feature",
                            orientation="h",
                            text="Importance",
                            labels={
                                "Importance": "Relative Importance",
                                "Feature": "Health Factor"
                            }
                        )

                        fig_importance.update_traces(
                            marker_color="#087E8B",
                            texttemplate="%{text:.3f}",
                            textposition="outside",
                            textfont=dict(
                                color="#102A43",
                                size=12
                            )
                        )

                        fig_importance.update_layout(
                            height=max(350, len(importance_df) * 45),
                            plot_bgcolor="white",
                            paper_bgcolor="white",
                            showlegend=False,
                            font=dict(
                                family="Arial",
                                color="#102A43",
                                size=13
                            ),
                            margin=dict(l=30, r=50, t=25, b=40),
                            xaxis=dict(
                                title="Relative Importance",
                                tickfont=dict(color="#102A43"),
                                gridcolor="#E2E8F0"
                            ),
                            yaxis=dict(
                                title="Health Factor",
                                tickfont=dict(color="#102A43")
                            )
                        )

                        with st.container(
                            key=f"feature_importance_{disease}"
                        ):
                            st.plotly_chart(
                                fig_importance,
                                use_container_width=True
                            )

                # -------------------------------------------------
                # DATASET INFORMATION
                # -------------------------------------------------

                dataset = details.get("dataset_info", {})

                if isinstance(dataset, dict) and dataset:

                    st.markdown("#### Dataset Information")

                    
                    total_samples = dataset.get("total_samples", 0)
                    positive_cases = dataset.get("positive_cases", 0)
                    negative_cases = dataset.get("negative_cases", 0)

                    dataset_cols = st.columns(4)

                    dataset_metrics = [
                        ("Total Samples", total_samples),
                        ("Positive Cases", positive_cases),
                        ("Negative Cases", negative_cases),
                        (
                            "Positive Rate",
                            f"{(positive_cases / total_samples * 100):.1f}%"
                            if total_samples else "N/A"
                        )
                    ]

                    for col, (label, value) in zip(
                        dataset_cols, dataset_metrics
                    ):
                        with col:
                            with st.container(border=True):
                                st.caption(label)
                                st.metric(
                                    label="",
                                    value=value
                                )

                    st.caption(
                        f"Dataset Source: {dataset.get('source', 'Not specified')}"
                    )

                    if dataset.get("note"):
                        st.info(dataset["note"])


            # -----------------------------------------------------
            # MEDICAL DISCLAIMER
            # -----------------------------------------------------

            st.markdown("---")

            st.warning(
                "⚠️ **Important Medical Disclaimer**\n\n"
                "These metrics represent performance on test datasets. "
                "They do not guarantee real-world diagnostic accuracy. "
                "Model performance can vary across different populations "
                "and clinical settings. This application is intended for "
                "educational purposes and should not replace professional "
                "medical advice."
            )

except requests.RequestException as e:

    st.error(f"Could not connect to the backend: {e}")

except (ValueError, TypeError) as e:

    st.error(f"Could not process model performance data: {e}")
