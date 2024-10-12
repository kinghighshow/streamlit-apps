
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("streamlit-apps/Expresso_churn_dataset_cleaned.csv")

# Assuming 'target' column contains the labels for whether a person has a bank account or not
X = df.drop(columns=['CHURN'])  # Features
Y = df['CHURN']  # Target labels


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
st.title("Customer Churn Prediction")

# App description
st.write("""
This app uses **eXtreame Gradient Boosting Classifier** to predict which individuals are most likely to churn from the Service Provider.
""")

# Display the dataset
st.write("### Xpresso Dataset Sample", df.head())

# Sidebar for user inputs
st.sidebar.header('User Input Parameters')

def user_input_features():
    MONTANT = st.sidebar.slider('Recharge Amount', int(df['MONTANT'].min()), int(df['MONTANT'].max()), int(df['MONTANT'].mean()))
    FREQUENCE_RECH = st.sidebar.slider('Recharge Frequency', int(df['FREQUENCE_RECH'].min()), int(df['FREQUENCE_RECH'].max()), int(df['FREQUENCE_RECH'].mean()))
    REVENUE = st.sidebar.slider('Revenue', int(df['REVENUE'].min()), int(df['REVENUE'].max()), int(df['REVENUE'].mean()))
    ARPU_SEGMENT = st.sidebar.slider('income over 90days/3', int(df['ARPU_SEGMENT'].min()), int(df['ARPU_SEGMENT'].max()), int(df['ARPU_SEGMENT'].mean()))
    FREQUENCE = st.sidebar.slider('Client monthly income', int(df['FREQUENCE'].min()), int(df['FREQUENCE'].max()), int(df['FREQUENCE'].mean()))
    REGULARITY = st.sidebar.slider('Regularity', int(df['REGULARITY'].min()), int(df['REGULARITY'].max()), int(df['REGULARITY'].mean()))
    Duration_of_use = st.sidebar.selectbox('Duration_of_use', ['Between_3to6', 'Between_6to9', 'Between_9to12', 'Between_12to15', 'Between_15to18', 'Between_18to21', 'Between_21to24', 'More_than_24'])
    
    data = {
        'MONTANT': MONTANT,
        'FREQUENCE_RECH': FREQUENCE_RECH,
        'REVENUE': REVENUE,
        'ARPU_SEGMENT': ARPU_SEGMENT,
        'FREQUENCE': FREQUENCE,
        'REGULARITY': REGULARITY,
        'Duration_of_use': Duration_of_use
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
st.write(f"Predicted Bank Account Status: {'Will churn' if prediction[0] == 1 else 'Will not churn'}")

st.subheader('Prediction Probability')
st.write(prediction_proba)

# Feature importance visualization
st.subheader('Feature Importance')
importance = gb_clf.feature_importances_
features = X.columns
plt.barh(features, importance)
st.pyplot(plt)
