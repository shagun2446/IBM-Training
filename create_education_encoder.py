import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv(r"C:\Users\shagu\OneDrive\Desktop\Training\Employee salary Project\Salary Data.csv")

# Create and fit encoder
le_education = LabelEncoder()
df['Education Level'] = le_education.fit_transform(df['Education Level'])

# Save encoder
pickle.dump(le_education, open('le_education.pkl', 'wb'))

print("✅ le_education.pkl file created successfully!")
