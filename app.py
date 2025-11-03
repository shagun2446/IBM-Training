import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# Streamlit Page Configuration
st.set_page_config(page_title="Salary Prediction App", layout="centered")
st.title(" Employee Salary Prediction App")
#st.write("Enter employee details below to predict their estimated salary:")

# Define Classes
gender_classes = ['Female', 'Male']
education_classes = ["Bachelor's", "Master's", 'PhD']
job_title_classes = ['Account Manager', 'Accountant', 'Administrative Assistant',
'Business Analyst', 'Business Development Manager',
'Business Intelligence Analyst', 'CEO', 'Chief Data Officer',
'Chief Technology Officer', 'Content Marketing Manager', 'Copywriter',
'Creative Director', 'Customer Service Manager', 'Customer Service Rep',
'Customer Service Representative', 'Customer Success Manager',
'Customer Success Rep', 'Data Analyst', 'Data Entry Clerk', 'Data Scientist',
'Digital Content Producer', 'Digital Marketing Manager', 'Director',
'Director of Business Development', 'Director of Engineering',
'Director of Finance', 'Director of HR', 'Director of Human Capital',
'Director of Human Resources', 'Director of Marketing',
'Director of Operations', 'Director of Product Management',
'Director of Sales', 'Director of Sales and Marketing', 'Event Coordinator',
'Financial Advisor', 'Financial Analyst', 'Financial Manager',
'Graphic Designer', 'HR Generalist', 'HR Manager', 'Help Desk Analyst',
'Human Resources Director', 'IT Manager', 'IT Support',
'IT Support Specialist', 'Junior Account Manager', 'Junior Accountant',
'Junior Advertising Coordinator', 'Junior Business Analyst',
'Junior Business Development Associate',
'Junior Business Operations Analyst', 'Junior Copywriter',
'Junior Customer Support Specialist', 'Junior Data Analyst',
'Junior Data Scientist', 'Junior Designer', 'Junior Developer',
'Junior Financial Advisor', 'Junior Financial Analyst',
'Junior HR Coordinator', 'Junior HR Generalist', 'Junior Marketing Analyst',
'Junior Marketing Coordinator', 'Junior Marketing Manager',
'Junior Marketing Specialist', 'Junior Operations Analyst',
'Junior Operations Coordinator', 'Junior Operations Manager',
'Junior Product Manager', 'Junior Project Manager', 'Junior Recruiter',
'Junior Research Scientist', 'Junior Sales Representative',
'Junior Social Media Manager', 'Junior Social Media Specialist',
'Junior Software Developer', 'Junior Software Engineer',
'Junior UX Designer', 'Junior Web Designer', 'Junior Web Developer',
'Marketing Analyst', 'Marketing Coordinator', 'Marketing Manager',
'Marketing Specialist', 'Network Engineer', 'Office Manager',
'Operations Analyst', 'Operations Director', 'Operations Manager',
'Principal Engineer', 'Principal Scientist', 'Product Designer',
'Product Manager', 'Product Marketing Manager', 'Project Engineer',
'Project Manager', 'Public Relations Manager', 'Recruiter',
'Research Director', 'Research Scientist', 'Sales Associate',
'Sales Director', 'Sales Executive', 'Sales Manager',
'Sales Operations Manager', 'Sales Representative',
'Senior Account Executive', 'Senior Account Manager', 'Senior Accountant',
'Senior Business Analyst', 'Senior Business Development Manager',
'Senior Consultant', 'Senior Data Analyst', 'Senior Data Engineer',
'Senior Data Scientist', 'Senior Engineer', 'Senior Financial Advisor',
'Senior Financial Analyst', 'Senior Financial Manager',
'Senior Graphic Designer', 'Senior HR Generalist', 'Senior HR Manager',
'Senior HR Specialist', 'Senior Human Resources Coordinator',
'Senior Human Resources Manager', 'Senior Human Resources Specialist',
'Senior IT Consultant', 'Senior IT Project Manager',
'Senior IT Support Specialist', 'Senior Manager',
'Senior Marketing Analyst', 'Senior Marketing Coordinator',
'Senior Marketing Director', 'Senior Marketing Manager',
'Senior Marketing Specialist', 'Senior Operations Analyst',
'Senior Operations Coordinator', 'Senior Operations Manager',
'Senior Product Designer', 'Senior Product Development Manager',
'Senior Product Manager', 'Senior Product Marketing Manager',
'Senior Project Coordinator', 'Senior Project Manager',
'Senior Quality Assurance Analyst', 'Senior Research Scientist',
'Senior Researcher', 'Senior Sales Manager', 'Senior Sales Representative',
'Senior Scientist', 'Senior Software Architect',
'Senior Software Developer', 'Senior Software Engineer',
'Senior Training Specialist', 'Senior UX Designer', 'Social Media Manager',
'Social Media Specialist', 'Software Developer', 'Software Engineer',
'Software Manager', 'Software Project Manager', 'Strategy Consultant',
'Supply Chain Analyst', 'Supply Chain Manager', 'Technical Recruiter',
'Technical Support Specialist', 'Technical Writer', 'Training Specialist',
'UX Designer', 'UX Researcher', 'VP of Finance', 'VP of Operations',
'Web Developer']


# Initialize Encoders
le_gender = LabelEncoder()
le_gender.classes_ = np.array(gender_classes)

le_education = LabelEncoder()
le_education.classes_ = np.array(education_classes)

le_job_title = LabelEncoder()
le_job_title.classes_ = np.array(job_title_classes)

# Load Model / Pipeline
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH = os.path.join(BASE_DIR, "salary_prediction_model.pkl")

if os.path.exists(PIPELINE_PATH):
    model_pipeline = joblib.load(PIPELINE_PATH)
    USE_PIPELINE = True
    st.info(" Using single pipeline (salary_prediction_model.pkl)")
else:
    model = joblib.load(os.path.join(BASE_DIR, "model_2.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    USE_PIPELINE = False

# Input Section
st.subheader("Employee Details")

age = st.number_input("Age", min_value=18, max_value=70, value=30)
gender = st.selectbox("Gender", gender_classes)
education = st.selectbox("Education Level", education_classes)
job_title = st.selectbox("Job Title", job_title_classes)
experience = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=5.0, step=0.5)

# Predict Salary
if st.button(" Predict Salary"):
    try:
        if USE_PIPELINE:
            input_df = pd.DataFrame([[age, gender, education, job_title, experience]],
                                    columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'])
            predicted_salary = model_pipeline.predict(input_df)[0]
        else:
            g = le_gender.transform([gender])[0]
            e = le_education.transform([education])[0]
            j = le_job_title.transform([job_title])[0]

            input_data = np.array([[g, e, j, experience, age]])
            input_scaled = scaler.transform(input_data)
            predicted_salary = model.predict(input_scaled)[0]

        st.success(f" Predicted Salary: ₹{predicted_salary:,.2f}")

    except Exception as e:
        st.error(f" Prediction Error: {e}")


