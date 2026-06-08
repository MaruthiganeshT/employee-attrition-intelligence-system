import streamlit as st
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt
import sys

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.explainer import get_explainer, generate_local_explanation
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)

# Page config
st.set_page_config(page_title="Employee Attrition Intelligence", page_icon="🏢", layout="wide")

# Custom CSS for rich aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #e6e6e6;
    }
    h1, h2, h3 {
        color: #00d4ff;
    }
    .metric-card {
        background: #1e2129;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        border-left: 5px solid #00d4ff;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏢 Employee Attrition Intelligence System")
st.markdown("Predict the probability of an employee leaving the company and understand the key driving factors.")

@st.cache_resource
def load_assets():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model = joblib.load(os.path.join(BASE_DIR, "models", "best_model.joblib"))
    preprocessor = joblib.load(os.path.join(BASE_DIR, "models", "preprocessor.joblib"))
    raw_df = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "WA_Fn-UseC_-HR-Employee-Attrition.csv"))
    X_train = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "X_train.csv"))
    
    # Pre-calculate medians and modes for defaults
    defaults = {}
    for col in raw_df.columns:
        if col not in ['Attrition', 'EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']:
            if raw_df[col].dtype == 'object':
                defaults[col] = raw_df[col].mode()[0]
            else:
                defaults[col] = raw_df[col].median()
                
    return model, preprocessor, X_train, defaults

try:
    model, preprocessor, X_train, defaults = load_assets()
except Exception as e:
    st.error("Assets not found. Please ensure the pipeline (preprocessing and modeling) has been run.")
    st.stop()

# Sidebar inputs
st.sidebar.header("Employee Profile Input")
st.sidebar.markdown("Adjust key features below:")

# Select the most important features to expose to the user, rest use defaults
age = st.sidebar.slider("Age", 18, 65, int(defaults.get('Age', 35)))
monthly_income = st.sidebar.number_input("Monthly Income", 1000, 20000, int(defaults.get('MonthlyIncome', 5000)))
job_satisfaction = st.sidebar.selectbox("Job Satisfaction (1-Low to 4-High)", [1, 2, 3, 4], index=int(defaults.get('JobSatisfaction', 3))-1)
over_time = st.sidebar.selectbox("Over Time", ['Yes', 'No'], index=1 if defaults.get('OverTime', 'No') == 'No' else 0)
distance_from_home = st.sidebar.slider("Distance From Home", 1, 30, int(defaults.get('DistanceFromHome', 10)))
years_at_company = st.sidebar.slider("Years At Company", 0, 40, int(defaults.get('YearsAtCompany', 5)))

# Construct user input dataframe
user_data = defaults.copy()
user_data['Age'] = age
user_data['MonthlyIncome'] = monthly_income
user_data['JobSatisfaction'] = job_satisfaction
user_data['OverTime'] = over_time
user_data['DistanceFromHome'] = distance_from_home
user_data['YearsAtCompany'] = years_at_company

input_df = pd.DataFrame([user_data])

# Feature Engineering step (must match preprocessing.py)

satisfaction_cols = [
    'EnvironmentSatisfaction',
    'JobSatisfaction',
    'RelationshipSatisfaction'
]

if all(col in input_df.columns for col in satisfaction_cols):
    input_df['TotalSatisfaction'] = (
        input_df[satisfaction_cols].sum(axis=1)
    )

# Career Stability
if {
    'YearsAtCompany',
    'TotalWorkingYears'
}.issubset(input_df.columns):

    input_df['CareerStability'] = (
        input_df['YearsAtCompany'] /
        (input_df['TotalWorkingYears'] + 1)
    )

# Income Per Year At Company
if {
    'MonthlyIncome',
    'YearsAtCompany'
}.issubset(input_df.columns):

    input_df['IncomePerYearAtCompany'] = (
        input_df['MonthlyIncome'] /
        (input_df['YearsAtCompany'] + 1)
    )

# Overtime Satisfaction Risk
if {
    'OverTime',
    'JobSatisfaction'
}.issubset(input_df.columns):

    overtime = input_df['OverTime'].map({
        'Yes': 1,
        'No': 0
    })

    input_df['OvertimeSatisfactionRisk'] = (
        overtime *
        (5 - input_df['JobSatisfaction'])
    )

# Preprocessing
processed_input = preprocessor.transform(input_df)

# Prediction
if st.button("Analyze Attrition Risk", type="primary"):
    with st.spinner("Analyzing profile..."):
        prob = model.predict_proba(processed_input)[0][1]
        
        # Determine risk category
        if prob > 0.6:
            risk = "High Risk"
            color = "#ff4b4b"
        elif prob > 0.3:
            risk = "Medium Risk"
            color = "#ffa100"
        else:
            risk = "Low Risk"
            color = "#00cc96"
            
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <h2 style="color: {color};">{prob:.1%}</h2>
                    <p>Attrition Probability</p>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <h2 style="color: {color};">{risk}</h2>
                    <p>Risk Category</p>
                </div>
            """, unsafe_allow_html=True)
            
        # SHAP Explainability
        st.info(
            f"The model estimates a "
            f"{prob:.1%} probability of attrition."
            )
        st.subheader("Key Driving Factors (Local SHAP Explanation)")
        try:
            # We need to pass the processed input as dataframe for SHAP
            # The column names were saved in X_train
            processed_input_df = pd.DataFrame(processed_input, columns=X_train.columns)
            
            explainer = get_explainer(model, X_train)
            sv, ev = generate_local_explanation(explainer, processed_input_df, X_train.columns)
            
            # Using matplotlib to generate waterfall/force plot
            fig, ax = plt.subplots(figsize=(10, 4))
            shap.waterfall_plot(shap.Explanation(values=sv, base_values=ev, data=processed_input_df.iloc[0], feature_names=X_train.columns), show=False)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not generate SHAP explanation: {e}")
    



st.markdown("### Global Attrition Insights")
metrics_path = os.path.join(
    BASE_DIR,
    "eda_reports",
    "model_metrics.csv"
)

if os.path.exists(metrics_path):

    st.subheader("Model Performance")

    metrics_df = pd.read_csv(
        metrics_path
    )

    st.dataframe(
        metrics_df,
        use_container_width=True
    )
top_features_path = os.path.join(
    BASE_DIR,
    "eda_reports",
    "top_features.csv"
)

if os.path.exists(top_features_path):

    st.subheader(
        "Top Global Drivers of Attrition"
    )

    top_features = pd.read_csv(
        top_features_path
    )

    st.dataframe(
        top_features.head(10),
        use_container_width=True,
        hide_index=True
    )

shap_summary_path = os.path.join(BASE_DIR, "eda_reports", "shap_summary.png")

if os.path.exists(shap_summary_path):
    st.image(shap_summary_path, caption="Global Feature Importance (SHAP Summary)")
else:
    st.info("Global SHAP summary plot not found. Run the explainer script to generate it.")
st.markdown("---")
st.caption(
    "Built with Scikit-Learn, XGBoost, SHAP and Streamlit."
)