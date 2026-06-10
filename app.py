import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Explainable Credit Risk Prediction",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #1f2937;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 25px;
    }

    .metric-card {
        background-color: #f9fafb;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .good-box {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        padding: 25px;
        border-radius: 18px;
        border-left: 8px solid #16a34a;
        color: #14532d;
        font-size: 20px;
        font-weight: 700;
    }

    .bad-box {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        padding: 25px;
        border-radius: 18px;
        border-left: 8px solid #dc2626;
        color: #7f1d1d;
        font-size: 20px;
        font-weight: 700;
    }

    .info-box {
        background-color: #eff6ff;
        padding: 18px;
        border-radius: 14px;
        border-left: 6px solid #2563eb;
        color: #1e3a8a;
    }

    .warning-box {
        background-color: #fffbeb;
        padding: 18px;
        border-radius: 14px;
        border-left: 6px solid #f59e0b;
        color: #78350f;
    }

    div.stButton > button {
        width: 100%;
        height: 3.2em;
        font-size: 18px;
        font-weight: 700;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "processed" / "german_credit_clean.csv"

MODEL_PATHS = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
    "Decision Tree": MODEL_DIR / "decision_tree.joblib",
    "HistGradientBoosting": MODEL_DIR / "histgradientboosting.joblib",
}

# Feature lists
numeric_features = [
    "duration_months",
    "credit_amount",
    "installment_rate",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents"
]

categorical_features = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment_since",
    "personal_status_sex",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker"
]

# User-friendly mappings
checking_status_map = {
    "A11": "Less than 0 DM",
    "A12": "0 to 200 DM",
    "A13": "Greater than or equal to 200 DM",
    "A14": "No checking account"
}

credit_history_map = {
    "A30": "No credits taken / all credits paid back",
    "A31": "All credits at this bank paid back",
    "A32": "Existing credits paid back duly",
    "A33": "Delay in paying off in the past",
    "A34": "Critical account / other existing credits"
}

purpose_map = {
    "A40": "Car - new",
    "A41": "Car - used",
    "A42": "Furniture/equipment",
    "A43": "Radio/television",
    "A44": "Domestic appliances",
    "A45": "Repairs",
    "A46": "Education",
    "A47": "Vacation",
    "A48": "Retraining",
    "A49": "Business",
    "A410": "Others"
}

savings_status_map = {
    "A61": "Less than 100 DM",
    "A62": "100 to 500 DM",
    "A63": "500 to 1000 DM",
    "A64": "Greater than or equal to 1000 DM",
    "A65": "Unknown / no savings account"
}

employment_since_map = {
    "A71": "Unemployed",
    "A72": "Less than 1 year",
    "A73": "1 to 4 years",
    "A74": "4 to 7 years",
    "A75": "Greater than 7 years"
}

personal_status_sex_map = {
    "A91": "Male - divorced/separated",
    "A92": "Female - divorced/separated/married",
    "A93": "Male - single",
    "A94": "Male - married/widowed",
    "A95": "Female - single"
}

other_debtors_map = {
    "A101": "None",
    "A102": "Co-applicant",
    "A103": "Guarantor"
}

property_map = {
    "A121": "Real estate",
    "A122": "Building society savings/life insurance",
    "A123": "Car or other property",
    "A124": "Unknown / no property"
}

other_installment_plans_map = {
    "A141": "Bank",
    "A142": "Stores",
    "A143": "None"
}

housing_map = {
    "A151": "Rent",
    "A152": "Own",
    "A153": "For free"
}

job_map = {
    "A171": "Unemployed / unskilled non-resident",
    "A172": "Unskilled resident",
    "A173": "Skilled employee / official",
    "A174": "Management / self-employed / highly qualified"
}

telephone_map = {
    "A191": "No telephone",
    "A192": "Telephone registered"
}

foreign_worker_map = {
    "A201": "Yes",
    "A202": "No"
}

def reverse_map(mapping, selected_label):
    for code, label in mapping.items():
        if label == selected_label:
            return code
    return list(mapping.keys())[0]

# Cached loading
@st.cache_resource
def load_model(model_name):
    path = MODEL_PATHS[model_name]
    return joblib.load(path)

@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None

# Helper functions
def create_input_dataframe(values):
    return pd.DataFrame([values], columns=numeric_features + categorical_features)

def make_gauge(probability_bad):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability_bad * 100,
            title={"text": "Bad Credit Risk Probability"},
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#ef4444" if probability_bad >= 0.5 else "#22c55e"},
                "steps": [
                    {"range": [0, 30], "color": "#dcfce7"},
                    {"range": [30, 60], "color": "#fef3c7"},
                    {"range": [60, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#111827", "width": 4},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(height=330, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def simple_risk_factors(applicant):
    factors = []

    if applicant["checking_status"] == "A11":
        factors.append(("High risk", "Checking account balance is less than 0 DM"))
    elif applicant["checking_status"] == "A14":
        factors.append(("Lower risk", "No checking account recorded"))

    if applicant["duration_months"] >= 36:
        factors.append(("High risk", "Long loan duration"))
    elif applicant["duration_months"] <= 12:
        factors.append(("Lower risk", "Short loan duration"))

    if applicant["credit_amount"] >= 7000:
        factors.append(("High risk", "Large credit amount"))
    elif applicant["credit_amount"] <= 2000:
        factors.append(("Lower risk", "Small credit amount"))

    if applicant["savings_status"] == "A61":
        factors.append(("High risk", "Low savings account balance"))
    elif applicant["savings_status"] in ["A64", "A65"]:
        factors.append(("Lower risk", "Strong or unknown/no savings status"))

    if applicant["credit_history"] in ["A33", "A34"]:
        factors.append(("Important", "Credit history needs attention"))

    if applicant["age"] < 25:
        factors.append(("Medium risk", "Young applicant"))
    elif applicant["age"] >= 45:
        factors.append(("Lower risk", "Older applicant"))

    return factors

# Header
st.markdown('<div class="main-title">💳 Explainable Credit Risk Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Interactive machine learning web app for predicting whether a loan applicant is a good or bad credit risk.</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-box">
    This app demonstrates an end-to-end machine learning workflow for credit-risk prediction using the German Credit dataset.
    It is designed for portfolio and educational purposes and should not be used for real lending decisions.
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# Sidebar
st.sidebar.title("⚙️ Model Settings")

selected_model_name = st.sidebar.selectbox(
    "Choose model",
    list(MODEL_PATHS.keys()),
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("About the prediction")
st.sidebar.write(
    "The selected model estimates whether an applicant is more likely to have lower or higher credit default risk based on financial and personal attributes."
)

st.sidebar.info(
    "Recommended default: Logistic Regression, based on the highest ROC-AUC in model comparison."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Tech Stack")
st.sidebar.write("Python · scikit-learn · Streamlit · Plotly · SHAP")

st.sidebar.markdown("---")
st.sidebar.subheader("Links")
st.sidebar.markdown(
    "[GitHub Repository](https://github.com/AbedalazizAbualhayjaa-hub/explainable-credit-risk-prediction)"
)

# Load model
try:
    model = load_model(selected_model_name)
except Exception as e:
    st.error(f"Could not load the selected model: {e}")
    st.stop()

df = load_data()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔮 Prediction", "📊 Dataset", "🧠 Model Performance", "📌 Project Overview"]
)

# Prediction tab
with tab1:
    st.subheader("Enter Applicant Information")
    st.markdown(
        "Adjust the applicant details below and click **Predict Credit Risk** to generate a prediction, probability score, and explanation."
    )

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Financial Information")

        duration_months = st.slider("Loan duration in months", 4, 72, 24)
        credit_amount = st.slider("Credit amount", 250, 20000, 3000, step=250)
        installment_rate = st.slider("Installment rate as % of disposable income", 1, 4, 2)
        existing_credits = st.slider("Number of existing credits", 1, 4, 1)

        checking_status_label = st.selectbox(
            "Checking account status",
            list(checking_status_map.values()),
            index=1
        )

        savings_status_label = st.selectbox(
            "Savings account status",
            list(savings_status_map.values()),
            index=0
        )

        credit_history_label = st.selectbox(
            "Credit history",
            list(credit_history_map.values()),
            index=2
        )

    with col_right:
        st.markdown("### Personal & Employment Information")

        age = st.slider("Age", 18, 80, 30)
        residence_since = st.slider("Years at current residence", 1, 4, 2)
        num_dependents = st.slider("Number of dependents", 1, 2, 1)

        purpose_label = st.selectbox(
            "Loan purpose",
            list(purpose_map.values()),
            index=3
        )

        employment_since_label = st.selectbox(
            "Employment duration",
            list(employment_since_map.values()),
            index=2
        )

        personal_status_sex_label = st.selectbox(
            "Personal status / sex",
            list(personal_status_sex_map.values()),
            index=2
        )

        housing_label = st.selectbox(
            "Housing",
            list(housing_map.values()),
            index=1
        )

    with st.expander("Advanced applicant details"):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            other_debtors_label = st.selectbox(
                "Other debtors",
                list(other_debtors_map.values()),
                index=0
            )

            property_label = st.selectbox(
                "Property",
                list(property_map.values()),
                index=0
            )

        with col_b:
            other_installment_plans_label = st.selectbox(
                "Other installment plans",
                list(other_installment_plans_map.values()),
                index=2
            )

            job_label = st.selectbox(
                "Job type",
                list(job_map.values()),
                index=2
            )

        with col_c:
            telephone_label = st.selectbox(
                "Telephone",
                list(telephone_map.values()),
                index=0
            )

            foreign_worker_label = st.selectbox(
                "Foreign worker",
                list(foreign_worker_map.values()),
                index=0
            )

    applicant = {
        "duration_months": duration_months,
        "credit_amount": credit_amount,
        "installment_rate": installment_rate,
        "residence_since": residence_since,
        "age": age,
        "existing_credits": existing_credits,
        "num_dependents": num_dependents,

        "checking_status": reverse_map(checking_status_map, checking_status_label),
        "credit_history": reverse_map(credit_history_map, credit_history_label),
        "purpose": reverse_map(purpose_map, purpose_label),
        "savings_status": reverse_map(savings_status_map, savings_status_label),
        "employment_since": reverse_map(employment_since_map, employment_since_label),
        "personal_status_sex": reverse_map(personal_status_sex_map, personal_status_sex_label),
        "other_debtors": reverse_map(other_debtors_map, other_debtors_label),
        "property": reverse_map(property_map, property_label),
        "other_installment_plans": reverse_map(other_installment_plans_map, other_installment_plans_label),
        "housing": reverse_map(housing_map, housing_label),
        "job": reverse_map(job_map, job_label),
        "telephone": reverse_map(telephone_map, telephone_label),
        "foreign_worker": reverse_map(foreign_worker_map, foreign_worker_label),
    }

    input_df = create_input_dataframe(applicant)

    st.write("")
    predict_button = st.button("🔍 Predict Credit Risk")

    if predict_button:
        prediction = model.predict(input_df)[0]

        if hasattr(model, "predict_proba"):
            probability_bad = model.predict_proba(input_df)[0][1]
        else:
            probability_bad = float(prediction)

        probability_good = 1 - probability_bad

        st.markdown("---")
        st.subheader("Prediction Result")

        result_col1, result_col2 = st.columns([1.1, 1])

        with result_col1:
            if prediction == 1:
                st.markdown(
                    """
                    <div class="bad-box">
                    ⚠️ Prediction: Higher Default Risk<br>
                    The applicant may have a higher probability of credit default.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div class="good-box">
                    ✅ Prediction: Lower Default Risk<br>
                    The applicant appears to have a lower probability of credit default.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.write("")
            m1, m2, m3 = st.columns(3)

            with m1:
                st.metric("Lower Risk Probability", f"{probability_good * 100:.1f}%")

            with m2:
                st.metric("Higher Risk Probability", f"{probability_bad * 100:.1f}%")

            with m3:
                risk_level = "Low" if probability_bad < 0.30 else "Medium" if probability_bad < 0.60 else "High"
                st.metric("Risk Level", risk_level)

        with result_col2:
            st.plotly_chart(make_gauge(probability_bad), use_container_width=True)

        st.markdown("---")
        st.subheader("Applicant Summary")

        summary_data = {
            "Feature": [
                "Loan Duration",
                "Credit Amount",
                "Age",
                "Checking Status",
                "Savings Status",
                "Employment",
                "Housing",
                "Purpose"
            ],
            "Value": [
                f"{duration_months} months",
                f"{credit_amount}",
                f"{age} years",
                checking_status_label,
                savings_status_label,
                employment_since_label,
                housing_label,
                purpose_label
            ]
        }

        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Simple Risk Explanation")

        factors = simple_risk_factors(applicant)

        if factors:
            for label, explanation in factors:
                if label == "High risk":
                    st.warning(f"⚠️ {explanation}")
                elif label == "Lower risk":
                    st.success(f"✅ {explanation}")
                else:
                    st.info(f"ℹ️ {explanation}")
        else:
            st.info("No major rule-based risk indicators were triggered.")

        st.markdown(
            """
            <div class="warning-box">
            Note: This explanation is a simplified interpretation layer for the web app.
            The actual model prediction comes from the trained machine learning pipeline.
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("View raw model input"):
            st.dataframe(input_df, use_container_width=True)

# Dataset overview tab
with tab2:
    st.subheader("Dataset Overview")

    if df is not None:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Rows", df.shape[0])

        with col2:
            st.metric("Columns", df.shape[1])

        with col3:
            if "target_binary" in df.columns:
                bad_rate = df["target_binary"].mean() * 100
                st.metric("Bad Credit Risk Rate", f"{bad_rate:.1f}%")

        st.markdown("### Sample Data")
        st.dataframe(df.head(10), use_container_width=True)

        if "target_binary" in df.columns:
            st.markdown("### Target Distribution")
            counts = df["target_binary"].map({0: "Lower Default Risk", 1: "Higher Default Risk"}).value_counts()
            fig = px.pie(
                names=counts.index,
                values=counts.values,
                title="Good vs Bad Credit Risk Distribution",
                hole=0.45
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Numeric Feature Distribution")
        selected_numeric = st.selectbox("Choose numeric feature", numeric_features)
        fig_hist = px.histogram(df, x=selected_numeric, nbins=30, title=f"Distribution of {selected_numeric}")
        st.plotly_chart(fig_hist, use_container_width=True)

    else:
        st.error("Dataset file was not found. Please check data/processed/german_credit_clean.csv.")

# Model info tab
with tab3:
    st.subheader("Model Information")

    st.markdown(
        f"""
        ### Selected Model: `{selected_model_name}`

        This app loads the trained model directly from the `models/` folder.

        The project originally trained and compared:

        - Logistic Regression
        - Decision Tree
        - Random Forest
        - HistGradientBoosting

        Logistic Regression is recommended as the default because it achieved the strongest ROC-AUC in your model comparison notebook.
        """
    )

    metrics_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "HistGradientBoosting"],
        "Accuracy": [0.750, 0.570, 0.755, 0.765],
        "Precision": [0.558, 0.377, 0.590, 0.644],
        "Recall": [0.800, 0.667, 0.600, 0.483],
        "F1-score": [0.658, 0.482, 0.595, 0.552],
        "ROC-AUC": [0.806, 0.599, 0.803, 0.797],
    })

    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    fig = px.bar(
        metrics_df,
        x="Model",
        y="ROC-AUC",
        title="Model Comparison by ROC-AUC",
        text="ROC-AUC"
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        ### Why ROC-AUC matters

        ROC-AUC measures how well the model separates good and bad credit-risk applicants across different classification thresholds.
        For credit-risk problems, this is useful because the threshold may be adjusted depending on business risk tolerance.
        """
    )

# How to use tab
with tab4:
    st.subheader("Project Overview")

    st.markdown(
        """
        This web app demonstrates an end-to-end machine learning workflow for credit-risk prediction.

        The user enters applicant information, and the trained model predicts whether the applicant is more likely to have lower or higher credit default risk.

        ### What this app demonstrates

        - Data preprocessing and feature engineering
        - Model training and comparison
        - Classification performance evaluation
        - Interactive machine learning deployment
        - User-friendly prediction interface
        - Basic explainability through risk-factor interpretation

        ### How to use the app

        1. Open the **Prediction** tab.
        2. Enter the applicant's financial and personal information.
        3. Click **Predict Credit Risk**.
        4. Review the prediction, probability score, risk level, and explanation.

        ### Important limitation

        This application is a portfolio demonstration of an end-to-end machine learning workflow. It is not intended for real financial lending decisions.
        """
    )