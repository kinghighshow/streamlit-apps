
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("streamlit-apps/financial_institution_cleaned1.csv")

# Assuming 'target' column contains the labels for whether a person has a bank account or not
X = df.drop(columns=['bank_account'])  # Features
Y = df['bank_account']  # Target labels


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the classifier
gb_clf = GradientBoostingClassifier(
    learning_rate=0.1,
    max_depth=4,
    min_samples_leaf=4,
    min_samples_split=5,
    n_estimators=100,
    subsample=1.0
)
gb_clf.fit(X_train_scaled, y_train)

# Streamlit app title
st.title("Financial Institution Classification")

# App description
st.write("""
This app uses **Gradient Boosting Classifier** to predict which individuals are most likely to have or use a bank account.
""")

# Display the dataset
st.write("### Financial Dataset Sample", df.head())

# Sidebar for user inputs
st.sidebar.header('User Input Parameters')

def user_input_features():
    country = st.sidebar.selectbox("country", ["country_Kenya", "country_Rwanda", "country_Tanzania", "country_Uganda"])
    location_type = st.sidebar.selectbox("location_type", ["location_type_Rural", "location_type_Urban"])
    cellphone_access = st.sidebar.selectbox('cellphone_access', ['cellphone_access_No', 'cellphone_access_Yes'])
    household_size = st.sidebar.slider('Household Size', int(df['household_size'].min()), int(df['household_size'].max()), int(df['household_size'].mean()))
    age_of_respondent = st.sidebar.slider('Respondent Age', int(df['age_of_respondent'].min()), int(df['age_of_respondent'].max()), int(df['age_of_respondent'].mean()))
    gender_of_respondent = st.sidebar.selectbox("gender_of_respondent", ['gender_of_respondent_Female', 'gender_of_respondent_Male'])
    relationship_with_head = st.sidebar.selectbox('relationship_with_head', ['relationship_with_head_Child', 'relationship_with_head_Head_of_Household', 'relationship_with_head_Other_non_relatives', 'relationship_with_head_Other_relative', 'relationship_with_head_Parent', 'relationship_with_head_Spouse'])
    marital_status = st.sidebar.selectbox('marital_status', ['marital_status_Divorced_Seperated', 'marital_status_Dont_know', 'marital_status_Married_Living_together', 'marital_status_Single_Never_Married', 'marital_status_Widowed'])
    education_level = st.sidebar.selectbox('education_level', ['education_level_No_formal_education', 'education_level_Other_Dont_know_RTA', 'education_level_Primary_education', 'education_level_Secondary_education', 'education_level_Tertiary_education', 'education_level_Vocational_Specialised_training'])
    job_type = st.sidebar.selectbox('job_type', ['job_type_Dont_Know_Refuse_to_answer', 'job_type_Farming_and_Fishing', 'job_type_Formally_employed_Government', 'job_type_Formally_employed_Private', 'job_type_Government_Dependent', 'job_type_Informally_employed', 'job_type_No_Income', 'job_type_Other_Income', 'job_type_Remittance_Dependent', 'job_type_Self_employed'])

    data = {
        'country': country,
        'location_type': location_type,
        'cellphone_access': cellphone_access,
        'household_size': household_size,
        'age_of_respondent': age_of_respondent,
        'gender_of_respondent': gender_of_respondent,
        'relationship_with_head': relationship_with_head,
        'marital_status': marital_status,
        'education_level': education_level,
        'job_type': job_type
    }

    # Convert the input data into a DataFrame
    features = pd.DataFrame(data, index=[0])
    
    # Align the features with the training data (adding missing columns if necessary)
    features = features.reindex(columns=X_train.columns, fill_value=0)
    
    return features

input_df = user_input_features()

# Make predictions based on user input
prediction = gb_clf.predict(input_df)
prediction_proba = gb_clf.predict_proba(input_df)

# Display the prediction results
st.subheader('Prediction')
st.write(f"Predicted Bank Account Status: {'Has Bank Account' if prediction[0] == 1 else 'No Bank Account'}")

st.subheader('Prediction Probability')
st.write(prediction_proba)

# Feature importance visualization
st.subheader('Feature Importance')
importance = gb_clf.feature_importances_
features = X.columns

plt.figure(figsize=(20, 14))
plt.barh(features, importance)
st.pyplot(plt)
