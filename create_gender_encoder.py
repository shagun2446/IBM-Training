import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv(r"C:\Users\shagu\OneDrive\Desktop\Training\Employee salary Project\Salary Data.csv")


# Create and fit encoder
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])

# Save encoder
pickle.dump(le_gender, open('le_gender.pkl', 'wb'))

print("✅ le_gender.pkl file created successfully!")
