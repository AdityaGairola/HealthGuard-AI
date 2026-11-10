# HealthGuard AI
### AI-Based Disease Risk Prediction and Health Monitoring System

HealthGuard AI is a machine learning-based health monitoring application designed to estimate the risk of three common diseases: **Diabetes, Heart Disease, and Hypertension**.

The system combines machine learning models with a web-based interface to provide risk predictions, personalized health recommendations, prediction history, and model performance analysis.

> **Disclaimer:** HealthGuard AI is an educational project and is not a medical diagnostic tool. Its predictions are not a substitute for professional medical advice, diagnosis, or treatment.

---

## 1. Project Objectives

- Develop an AI-based disease risk prediction system.
- Apply machine learning algorithms to healthcare datasets.
- Provide users with an interactive health monitoring dashboard.
- Display prediction probabilities and risk categories.
- Generate personalized health recommendations.
- Evaluate and compare machine learning model performance.

## 2. Key Features

### User Authentication
- User registration and login.
- Secure password hashing.
- Token-based authentication.

### Health Profile Management
- Maintain personal health information.
- Record health-related parameters.
- Store family medical history.

### Disease Risk Prediction
The application supports three prediction modules:

- **Diabetes Prediction:** Estimates diabetes risk using health parameters such as glucose, BMI, age, insulin and blood pressure.
- **Heart Disease Prediction:** Estimates heart disease risk using cardiovascular health indicators.
- **Hypertension Prediction:** Estimates hypertension risk using factors such as age, BMI, smoking, physical activity and family history.

### Interactive Dashboard
- View previous prediction records.
- Monitor prediction history.
- Review disease risk categories and model scores.

### Personalized Recommendations
- Displays health-related recommendations based on prediction results.
- Provides general lifestyle guidance.

### Model Performance Analysis
- Accuracy, precision, recall and F1-score.
- ROC-AUC evaluation.
- Confusion matrices.
- Model comparison.
- Feature importance visualization.

### PDF Reports
- Generate downloadable health prediction reports.

## 3. Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python 3.11 |
| Frontend | Streamlit |
| Backend | FastAPI |
| Machine Learning | Scikit-learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Model Explainability | SHAP |
| Data Visualization | Plotly |
| Database | SQLite, SQLAlchemy |
| Authentication | JWT, Passlib |
| Testing | Pytest |
| Report Generation | ReportLab |

## 4. System Architecture

The application follows a frontend-backend-machine learning architecture.

```text
             USER
               |
               v
       STREAMLIT FRONTEND
               |
               v
          FASTAPI
            BACKEND
               |
       +-------+-------+
       |       |       |
       v       v       v
    AUTH    PROFILE  PREDICTION
                       |
                       v
                ML PREDICTION
                   SERVICE
                       |
          +------------+------------+
          |            |            |
          v            v            v
       DIABETES      HEART       HYPERTENSION
        MODEL        MODEL         MODEL
          |            |            |
          +------------+------------+
                       |
                       v
               PREDICTION RESULT
                       |
          +------------+------------+
          |            |            |
          v            v            v
       DATABASE   DASHBOARD   RECOMMENDATIONS
```

## 5. Machine Learning Models

The project uses multiple machine learning algorithms and selects models based on cross-validation F1-score.

Algorithms explored include:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- Decision Tree
- XGBoost

The selected models are saved as serialized `.pkl` files and loaded by the backend prediction service.

### Selected Models

| Disease | Selected Model |
|---|---|
| Diabetes | Random Forest |
| Heart Disease | Logistic Regression |
| Hypertension | Logistic Regression |

## 6. Model Performance

The following results are from the project's saved evaluation reports.

| Metric | Diabetes | Heart Disease | Hypertension |
|---|---:|---:|---:|
| Accuracy | 83.12% | 60.66% | 72.08% |
| Precision | 79.17% | 41.18% | 17.11% |
| Recall | 70.37% | 77.78% | 76.47% |
| F1-Score | 74.51% | 53.85% | 27.96% |
| ROC-AUC | 87.31% | 70.03% | 75.97% |

**Important observations:**

- The diabetes model achieved the strongest overall performance.
- The heart disease model achieved 77.78% recall but had a relatively high false-positive rate.
- The hypertension model achieved 76.47% recall, but its low precision indicates many false-positive predictions.

The hypertension dataset is synthetically generated for demonstration and is not clinically validated. The evaluation results should not be interpreted as evidence of real-world diagnostic accuracy.

## 7. Dataset Information

The project contains three datasets:

| Dataset | Description |
|---|---|
| `diabetes.csv` | Pima Indians Diabetes dataset |
| `heart.csv` | UCI Cleveland Heart Disease dataset |
| `hypertension.csv` | Synthetic hypertension demonstration dataset |

The hypertension dataset was generated using assumed risk-factor distributions and a programmed labeling formula.

All datasets are intended for educational machine learning experimentation.

## 8. Project Structure

```text
HealthGuard-AI/
│
├── backend/
│   ├── authentication/
│   ├── database/
│   ├── routes/
│   ├── schemas/
│   ├── services/
│   └── main.py
│
├── frontend/
│   ├── components/
│   ├── pages/
│   │   ├── 01_home.py
│   │   ├── 02_auth.py
│   │   ├── 03_profile.py
│   │   ├── 04_prediction.py
│   │   ├── 05_dashboard.py
│   │   ├── 06_recommendations.py
│   │   └── 07_model_performance.py
│   ├── styles/
│   └── app.py
│
├── ml/
│   ├── datasets/
│   ├── evaluation/
│   ├── models/
│   ├── preprocessing/
│   └── training/
│
├── tests/
│   ├── test_api.py
│   └── test_prediction_services.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## 9. Installation and Setup

### Prerequisites

- Python 3.11
- Git
- Visual Studio Code (recommended)

### Step 1: Clone the repository

```bash
git clone https://github.com/AdityaGairola/HealthGuard-AI.git
cd HealthGuard-AI
```

### Step 2: Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure environment variables

Create a local `.env` file by copying the example:

```powershell
Copy-Item .env.example .env
```

Update the `SECRET_KEY` in `.env` with a securely generated random value.

The default database configuration uses SQLite.

### Step 5: Start the backend

Open a terminal in the project directory:

```powershell
uvicorn backend.main:app --reload
```

Backend API:

`http://127.0.0.1:8000`

Interactive API documentation:

`http://127.0.0.1:8000/docs`

### Step 6: Start the frontend

Open a second PowerShell terminal, activate the virtual environment, and run:

```powershell
streamlit run frontend/app.py
```

The Streamlit application will normally open at:

`http://localhost:8501`

**Important:** Keep the backend terminal running while using the frontend.

## 10. Running Automated Tests

The project includes automated API and machine learning prediction service tests.

Run:

```powershell
python -m pytest -v
```

Latest test results:

**7 tests passed.**

The tests cover:
- Root API endpoint.
- Health endpoint.
- Prediction route registration.
- OpenAPI documentation.
- Diabetes prediction service.
- Heart disease prediction service.
- Hypertension prediction service.

## 11. Future Enhancements

Potential improvements include:

- Integration with real-time wearable health monitoring devices.
- Improved hypertension prediction using validated clinical datasets.
- More extensive model validation and hyperparameter optimization.
- Additional disease prediction modules.
- Enhanced explainable AI visualizations.
- Cloud deployment and scalable database integration.
- More comprehensive automated integration testing.

## 12. Limitations

- Model performance depends on the quality and size of the available datasets.
- The hypertension dataset is synthetic and not clinically validated.
- Small evaluation datasets can produce unstable performance estimates.
- Predictions are intended for educational demonstration only.
- The application has not been validated for clinical use.

## 13. Author

**Aditya Gairola**

GitHub: [AdityaGairola](https://github.com/AdityaGairola)

Project Repository: [HealthGuard-AI](https://github.com/AdityaGairola/HealthGuard-AI)

---

## License

This project is developed for educational and academic purposes. No license has currently been specified.