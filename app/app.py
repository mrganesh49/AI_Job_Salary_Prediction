
import csv
from datetime import datetime

import streamlit as st
import pandas as pd
import json
import joblib
from pathlib import Path

# ==========================================
# 1. LOAD MODEL AND METADATA
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "final_job_salary_model.pkl"
METADATA_PATH = BASE_DIR / "models" / "input_metadata.json"

model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r") as file:
    metadata = json.load(file)

categorical_options = metadata["categorical_options"]
numerical_defaults = metadata["numerical_defaults"]


# ==========================================
# 2. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Job Salary Prediction",
    page_icon="💼",
    layout="wide"
)

st.title("💼 AI Job Salary Prediction")
st.write(
    "Predict the estimated annual salary based on job-related information."
)

st.info("Model: Optimized Gradient Boosting Regressor")


# ==========================================
# 3. USER INPUTS
# ==========================================

st.header("Job Information")

col1, col2 = st.columns(2)

with col1:

    job_title = st.selectbox(
        "Job Title",
        categorical_options["job_title"]
    )

    experience_level = st.selectbox(
        "Experience Level",
        categorical_options["experience_level"]
    )

    employment_type = st.selectbox(
        "Employment Type",
        categorical_options["employment_type"]
    )

    company_location = st.selectbox(
        "Company Location",
        categorical_options["company_location"]
    )

    company_size = st.selectbox(
        "Company Size",
        categorical_options["company_size"]
    )

    employee_residence = st.selectbox(
        "Employee Residence",
        categorical_options["employee_residence"]
    )

with col2:

    education_required = st.selectbox(
        "Education Required",
        categorical_options["education_required"]
    )

    industry = st.selectbox(
        "Industry",
        categorical_options["industry"]
    )

    remote_ratio = st.selectbox(
        "Remote Ratio",
        [0, 50, 100]
    )

    years_experience = st.number_input(
        "Years of Experience",
        min_value=0,
        max_value=50,
        value=max(0, min(50, round(
            numerical_defaults["years_experience"]
        )))
    )

    job_description_length = st.number_input(
        "Job Description Length",
        min_value=500,
        max_value=2499,
        value=max(500, min(2499, round(
            numerical_defaults["job_description_length"]
        )))
    )

    benefits_score = st.number_input(
        "Benefits Score",
        min_value=5.0,
        max_value=10.0,
        value=max(5.0, min(10.0,
            numerical_defaults["benefits_score"]
        )),
        step=0.1
    )


# ==========================================
# 4. ADDITIONAL INPUTS
# ==========================================

st.header("Additional Information")

col3, col4 = st.columns(2)

with col3:

    posting_year = st.number_input(
        "Posting Year",
        min_value=2024,
        max_value=2025,
        value=max(2024, min(2025, round(
            numerical_defaults["posting_year"]
        )))
    )

    posting_month = st.number_input(
        "Posting Month",
        min_value=1,
        max_value=12,
        value=max(1, min(12, round(
            numerical_defaults["posting_month"]
        )))
    )

    skill_count = st.number_input(
        "Number of Required Skills",
        min_value=3,
        max_value=5,
        value=max(3, min(5, round(
            numerical_defaults["skill_count"]
        )))
    )

with col4:

    requires_python = st.checkbox("Requires Python")

    requires_sql = st.checkbox("Requires SQL")


# ==========================================
# 5. FEATURE ENGINEERING
# ==========================================

def generate_categories(
    years,
    remote,
    benefits,
    description
):

    experience_category = pd.cut(
        [years],
        bins=[-1, 2, 5, 10, 15, float("inf")],
        labels=["Entry", "Junior", "Mid", "Senior", "Expert"]
    )[0]

    remote_category = {
        0: "On-site",
        50: "Hybrid",
        100: "Remote"
    }[remote]

    benefits_category = pd.cut(
        [benefits],
        bins=[5, 6.5, 8.5, 10],
        labels=["Low", "Medium", "High"],
        include_lowest=True
    )[0]

    description_category = pd.cut(
        [description],
        bins=[499, 999, 1999, 2499],
        labels=["Short", "Medium", "Long"]
    )[0]

    return (
        str(experience_category),
        remote_category,
        str(benefits_category),
        str(description_category)
    )


# ==========================================
# 6. SALARY PREDICTION
# ==========================================

if st.button("Predict Salary", type="primary"):

    exp_cat, remote_cat, benefits_cat, description_cat = (
        generate_categories(
            years_experience,
            remote_ratio,
            benefits_score,
            job_description_length
        )
    )

    input_data = {

        "job_title": job_title,
        "experience_level": experience_level,
        "employment_type": employment_type,
        "company_location": company_location,
        "company_size": company_size,
        "employee_residence": employee_residence,

        "remote_ratio": remote_ratio,
        "education_required": education_required,
        "years_experience": years_experience,
        "industry": industry,
        "job_description_length": job_description_length,
        "benefits_score": benefits_score,
        "posting_year": posting_year,
        "posting_month": posting_month,
        "skill_count": skill_count,
        "requires_python": int(requires_python),
        "requires_sql": int(requires_sql),

        "experience_category": exp_cat,
        "remote_work_category": remote_cat,
        "benefits_category": benefits_cat,
        "description_category": description_cat
    }

    input_df = pd.DataFrame([input_data])

    # Maintain the exact training feature order
    input_df = input_df[[
        "job_title",
        "experience_level",
        "employment_type",
        "company_location",
        "company_size",
        "employee_residence",
        "remote_ratio",
        "education_required",
        "years_experience",
        "industry",
        "job_description_length",
        "benefits_score",
        "posting_year",
        "posting_month",
        "skill_count",
        "requires_python",
        "requires_sql",
        "experience_category",
        "remote_work_category",
        "benefits_category",
        "description_category"
    ]]

    prediction = model.predict(input_df)[0]

# ==========================================
# 7. LOG PREDICTIONS
# ==========================================

    LOG_DIR = BASE_DIR / "logs"
    LOG_DIR.mkdir(exist_ok=True)

    LOG_FILE = LOG_DIR / "predictions.csv"

    log_data = input_data.copy()

    log_data["prediction_timestamp"] = datetime.now().isoformat(
        timespec="seconds"
    )

    log_data["predicted_salary_usd"] = round(float(prediction), 2)

    # Append prediction to CSV file
    file_exists = LOG_FILE.exists() and LOG_FILE.stat().st_size > 0

    pd.DataFrame([log_data]).to_csv(
        LOG_FILE,
        mode="a",
        header=not file_exists,
        index=False
    )

    st.success("Prediction recorded successfully!")


    st.success("Salary prediction completed!")

    st.subheader("Estimated Annual Salary")

    st.metric(
        "Predicted Salary (USD)",
        f"${prediction:,.2f}"
    )

    st.caption(
        "This is an estimate generated from the patterns "
        "learned by the trained model."
    )
