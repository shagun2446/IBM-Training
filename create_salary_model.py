'''import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

# -------------------------------
# Step 1: Load dataset
# -------------------------------
df = pd.read_csv(r"C:\Users\shagu\OneDrive\Desktop\Training\Employee salary Project\Salary Data.csv")

# If needed, rename columns to match these names:
# df.rename(columns={
#     'gender_column': 'Gender',
#     'education_column': 'Education Level',
#     'job_column': 'Job Title',
#     'salary_column': 'Salary'
# }, inplace=True)

# -------------------------------
# Step 2: Handle missing values
# -------------------------------
# Remove rows where Salary is missing
df = df.dropna(subset=['Salary'])

# Optionally fill missing in other columns (if any)
#df = df.fillna(method='ffill')
df = df.ffill()
# -------------------------------
# Step 3: Encode categorical columns
# -------------------------------
le_gender = LabelEncoder()
df["Gender"] = le_gender.fit_transform(df["Gender"])

le_education = LabelEncoder()
df["Education Level"] = le_education.fit_transform(df["Education Level"])

le_job = LabelEncoder()
df["Job Title"] = le_job.fit_transform(df["Job Title"])

# -------------------------------
# Step 4: Split features & target
# -------------------------------
X = df[["Gender", "Education Level", "Job Title"]]
y = df["Salary"]

# -------------------------------
# Step 5: Scale and train
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

# -------------------------------
# Step 6: Save model & encoders
# -------------------------------
joblib.dump(model, "salary_prediction_model.pkl")
joblib.dump(le_gender, "le_gender.pkl")
joblib.dump(le_education, "le_education.pkl")
joblib.dump(le_job, "le_job.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Salary prediction model and encoders saved successfully!")
'''
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

# 1️⃣ Load dataset
df = pd.read_csv("salary_data.csv")

# 2️⃣ Handle missing values
df = df.dropna(subset=['Salary'])  # Remove rows with missing salary
df = df.ffill()  # Fill missing other columns

# 3️⃣ Encode categorical columns
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])

le_education = LabelEncoder()
df['Education Level'] = le_education.fit_transform(df['Education Level'])

le_job = LabelEncoder()
df['Job Title'] = le_job.fit_transform(df['Job Title'])

# 4️⃣ Features and target
X = df[['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']]
y = df['Salary']

# 5️⃣ Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6️⃣ Train model
model = LinearRegression()
model.fit(X_scaled, y)

# 7️⃣ Save model and encoders
joblib.dump(model, "salary_prediction_model.pkl")
joblib.dump(le_gender, "le_gender.pkl")
joblib.dump(le_education, "le_education.pkl")
joblib.dump(le_job, "le_job_title.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Salary prediction model and encoders saved successfully!")
