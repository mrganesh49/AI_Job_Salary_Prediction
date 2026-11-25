
import joblib
import json

# Load the saved ML pipeline
model = joblib.load("models/final_job_salary_model.pkl")

# Access preprocessing steps
preprocessor = model.named_steps["preprocessor"]

encoder = preprocessor.named_transformers_["categorical"]
scaler = preprocessor.named_transformers_["numerical"]

# Original input column names
categorical_columns = [
    "job_title",
    "experience_level",
    "employment_type",
    "company_location",
    "company_size",
    "employee_residence",
    "education_required",
    "industry"
]

numerical_columns = [
    "remote_ratio",
    "years_experience",
    "job_description_length",
    "benefits_score",
    "posting_year",
    "posting_month",
    "skill_count",
    "requires_python",
    "requires_sql"
]

# Extract learned categorical options
categorical_options = {
    col: sorted(values.astype(str).tolist())
    for col, values in zip(categorical_columns, encoder.categories_)
}

# Use training-data means as numerical defaults
numerical_defaults = {
    col: float(value)
    for col, value in zip(numerical_columns, scaler.mean_)
}

# Create metadata
metadata = {
    "categorical_options": categorical_options,
    "numerical_defaults": numerical_defaults
}

# Save JSON file
with open("models/input_metadata.json", "w") as file:
    json.dump(metadata, file, indent=4)

print("Metadata created successfully!")
print("Categorical fields:", len(categorical_options))
print("Numerical fields:", len(numerical_defaults))
