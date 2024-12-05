import streamlit as st
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the trained SVR model
with open('svr_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Expected features based on the training dataset
expected_features = [
    'Age', 
    'Education_Bachelor’s degree', 
    'Education_Master’s degree', 
    'Education_Doctoral degree', 
    'ExperienceLevel_Entry level', 
    'ExperienceLevel_Mid-Senior level', 
    'ExperienceLevel_Executive'
]

# Streamlit App
def app():
    st.set_page_config(page_title="Job Level Predictor", layout="wide")

    
    st.sidebar.header("Input Features")
    st.sidebar.markdown("Enter the features below:")

    # User Inputs
    age = st.sidebar.slider("Age", 18, 75, 30)
    education_level = st.sidebar.selectbox(
        "Education Level", [
            "Bachelor’s degree", 
            "Master’s degree", 
            "Doctoral degree"
        ]
    )
    experience_level = st.sidebar.selectbox(
        "Experience Level", [
            "Entry level", 
            "Mid-Senior level", 
            "Executive"
        ]
    )

    # One-hot encode education and experience level
    input_data = pd.DataFrame({
        'Age': [age],
        'Education_Bachelor’s degree': [1 if education_level == "Bachelor’s degree" else 0],
        'Education_Master’s degree': [1 if education_level == "Master’s degree" else 0],
        'Education_Doctoral degree': [1 if education_level == "Doctoral degree" else 0],
        'ExperienceLevel_Entry level': [1 if experience_level == "Entry level" else 0],
        'ExperienceLevel_Mid-Senior level': [1 if experience_level == "Mid-Senior level" else 0],
        'ExperienceLevel_Executive': [1 if experience_level == "Executive" else 0],
    })

    # Ensure all expected features are present
    for feature in expected_features:
        if feature not in input_data.columns:
            input_data[feature] = 0

    # Reorder columns to match training
    input_data = input_data[expected_features]

    # Display input data
    st.markdown("### Input Data")
    st.dataframe(input_data)

    # Scale the input data
    try:
        scaled_data = scaler.transform(input_data)
        st.success("Input data scaled successfully.")
    except Exception as e:
        st.error(f"Error scaling input data: {e}")
        return

    # Predict button
    if st.button("Predict Job Level"):
        try:
            # Predict using the loaded model
            prediction = model.predict(scaled_data)
            st.markdown(f"### Predicted Job Level: {round(prediction[0], 2)}")

            # Visualization of contributions
            visualize_contributions(input_data, prediction[0])
        except Exception as e:
            st.error(f"Error making prediction: {e}")


# Visualization function
def visualize_contributions(input_data, prediction):
    # Mock calculation of contributions
    age_contribution = input_data['Age'].iloc[0] / 75
    education_contribution = (
        input_data['Education_Bachelor’s degree'].iloc[0] * 0.3 +
        input_data['Education_Master’s degree'].iloc[0] * 0.6 +
        input_data['Education_Doctoral degree'].iloc[0] * 1.0
    )
    experience_contribution = (
        input_data['ExperienceLevel_Entry level'].iloc[0] * 0.2 +
        input_data['ExperienceLevel_Mid-Senior level'].iloc[0] * 0.5 +
        input_data['ExperienceLevel_Executive'].iloc[0] * 0.8
    )

    # Normalize contributions
    total_contribution = age_contribution + education_contribution + experience_contribution
    age_contribution /= total_contribution
    education_contribution /= total_contribution
    experience_contribution /= total_contribution

    # Data for visualization
    contributions = {
        "Age Contribution": age_contribution,
        "Education Contribution": education_contribution,
        "Experience Contribution": experience_contribution,
        "Predicted Job Level": prediction
    }

    # Bar plot
    st.markdown("### Feature Contributions to Predicted Job Level")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=list(contributions.keys()), y=list(contributions.values()), palette="viridis", ax=ax)
    ax.set_title("Feature Contributions")
    ax.set_ylabel("Contribution Value")
    ax.set_xlabel("Features")

    # Annotate the bars with their values
    for i, value in enumerate(contributions.values()):
        ax.text(i, value + 0.01, f"{value:.2f}", ha='center', fontsize=10)

    st.pyplot(fig)


if __name__ == "__main__":
    app()
