# AI Job Salary Prediction

## Project Overview
An end-to-end Machine Learning project that predicts the estimated annual salary (USD) for a job based on job-related information.

## Problem Statement
Salary estimation can help understand how factors such as job title, experience, location, company size, and required skills relate to compensation.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Joblib
- Git and GitHub

## Machine Learning
- Models evaluated: Linear Regression, Decision Tree, Random Forest, Gradient Boosting
- Final model: Optimized Gradient Boosting Regressor
- Evaluation metrics:
  - MAE: $14,691.05
  - RMSE: $20,290.09
  - R² Score: 0.8871

## Project Structure
```text
AI_Job_Salary_Prediction/
├── app/
│   └── app.py
├── logs/
│   └── predictions.csv
├── models/
│   ├── final_job_salary_model.pkl
│   └── input_metadata.json
├── .gitignore
├── requirements.txt
└── README.md
```

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Output
The application accepts job-related inputs and predicts the estimated annual salary in USD.