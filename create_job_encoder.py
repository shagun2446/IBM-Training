import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv(r"C:\Users\shagu\OneDrive\Desktop\Training\Employee salary Project\Salary Data.csv")

# Create and fit encoder
le_job = LabelEncoder()
df['Job Title'] = le_job.fit_transform(df['Job Title'])

# Save encoder
pickle.dump(le_job, open('le_job_title.pkl', 'wb'))

print("✅ le_job_title.pkl file created successfully!")
