
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor

# -----------------------------
# Step 1: Load the CSV dataset
# -----------------------------
data_path = r"C:\Users\shagu\OneDrive\Desktop\Training\Employee salary Project\salary_data.csv"
df = pd.read_csv(data_path)

# -----------------------------
# Step 2: Fill missing values
# -----------------------------
df = df.ffill()  # forward fill for any missing data

# -----------------------------
# Step 3: Encode categorical columns
# -----------------------------
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])

le_education = LabelEncoder()
df['Education Level'] = le_education.fit_transform(df['Education Level'])

le_job_title = LabelEncoder()
df['Job Title'] = le_job_title.fit_transform(df['Job Title'])

# -----------------------------
# Step 4: Features and target
# -----------------------------
X = df[['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']]
y = df['Salary']

# -----------------------------
# Step 5: Scale features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Step 6: Train the model
# -----------------------------
model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# -----------------------------
# Step 7: Save model and encoders
# -----------------------------
joblib.dump(model, 'model_2.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_gender, 'le_gender.pkl')
joblib.dump(le_education, 'le_education.pkl')
joblib.dump(le_job_title, 'le_job_title.pkl')

print("✅ Salary prediction model and encoders saved successfully!")
