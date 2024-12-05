import streamlit as st
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the trained SVR model and scaler
with open('svr_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define the column order used during training
TRAINED_COLUMNS = [
    'Age',
    'ExperienceLevel_Director',
    'ExperienceLevel_Entry level',
    'ExperienceLevel_Executive',
    'ExperienceLevel_Internship',
    'ExperienceLevel_Mid-Senior level',
    'Education_Doctoral degree',
    'Education_I prefer not to answer',
    'Education_Master’s degree',
    'Education_No formal education past high school',
    'Education_Professional degree',
    'Education_Some college/university study without earning a bachelor’s degree'
]

def plot_feature_contributions(features, contributions):
    """
    Creates a bar plot to visualize feature contributions.

    Args:
    - features: List of feature names.
    - contributions: List of feature contribution values.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x=features, y=contributions, palette="viridis")
    plt.title("Feature Contributions to Predicted Job Level")
    plt.ylabel("Contribution Value")
    plt.xlabel("Features")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(plt)

def app():
    st.set_page_config(page_title="Job Level Predictor", layout="wide")

    # Add the heading question
    st.title("Can we accurately predict an individual's job level within an organization based on demographic and professional characteristics such as age, experience level, and education?")
    
    st.sidebar.header("Input Features")
    st.sidebar.markdown("Enter the features below:")

    # User Inputs
    age = st.sidebar.slider("Age", 18, 75, 30)
    education_level = st.sidebar.selectbox(
        "Education Level", [
            "Doctoral degree",
            "I prefer not to answer",
            "Master’s degree",
            "No formal education past high school",
            "Professional degree",
            "Some college/university study without earning a bachelor’s degree"
        ]
    )
    experience_level = st.sidebar.selectbox(
        "Experience Level", [
            "Director",
            "Entry level",
            "Executive",
            "Internship",
            "Mid-Senior level"
        ]
    )

    # One-hot encode the categorical inputs
    input_data = pd.DataFrame(columns=TRAINED_COLUMNS)
    input_data.loc[0] = 0  # Initialize all columns to 0

    # Set the values for the user input
    input_data['Age'] = age
    input_data[f'Education_{education_level}'] = 1
    input_data[f'ExperienceLevel_{experience_level}'] = 1

    # Display input data
    st.write("### Input Data")
    st.write(input_data)

    # Scale the input data
    try:
        # Pass the raw numpy array to avoid feature name issues
        scaled_data = scaler.transform(input_data.values)
        st.write("### Scaled Data")
        st.write(scaled_data)

        # Predict button
        if st.button("Predict Job Level"):
            prediction = model.predict(scaled_data)
            st.write(f"### Predicted Job Level: {round(prediction[0], 2)}")

            # Calculate and display feature contributions
            feature_contributions = scaled_data[0] * model.coef_ if hasattr(model, 'coef_') else scaled_data[0]
            plot_feature_contributions(TRAINED_COLUMNS, feature_contributions)
    except Exception as e:
        st.error(f"Error scaling input data: {e}")

if __name__ == "__main__":
    app()
