import streamlit as st
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the trained SVR model
try:
    with open('svr_model.pkl', 'rb') as file:
        model = pickle.load(file)
    st.write("✅ SVR Model loaded successfully.")
except FileNotFoundError:
    st.error("❌ The SVR model file (svr_model.pkl) is missing. Please upload it.")
    st.stop()

# Load the scaler
try:
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    st.write("✅ Scaler loaded successfully.")
except FileNotFoundError:
    st.error("❌ The scaler file (scaler.pkl) is missing. Please upload it.")
    st.stop()

# Streamlit App
def app():
    st.title("Job Level Predictor (SVR)")

    # Sidebar Inputs
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
    education_mapping = {
        "Bachelor’s degree": "Education_Bachelor’s degree",
        "Master’s degree": "Education_Master’s degree",
        "Doctoral degree": "Education_Doctoral degree"
    }
    experience_mapping = {
        "Entry level": "ExperienceLevel_Entry level",
        "Mid-Senior level": "ExperienceLevel_Mid-Senior level",
        "Executive": "ExperienceLevel_Executive"
    }

    # Create a placeholder DataFrame with all features used during training
    input_data = {
        'Age': age,
        'Education_Bachelor’s degree': 0,
        'Education_Master’s degree': 0,
        'Education_Doctoral degree': 0,
        'ExperienceLevel_Entry level': 0,
        'ExperienceLevel_Mid-Senior level': 0,
        'ExperienceLevel_Executive': 0,
    }

    # Set the selected options to 1 in the input_data
    input_data[education_mapping[education_level]] = 1
    input_data[experience_mapping[experience_level]] = 1

    # Convert the dictionary to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Display raw input data
    st.write("### Input Data")
    st.write(input_df)

    # Scale the input data
    try:
        scaled_data = scaler.transform(input_df)
    except Exception as e:
        st.error(f"Error scaling input data: {e}")
        st.stop()

    # Display scaled input data
    st.write("### Input Data (Scaled)")
    st.write(scaled_data)

    # Visualization Function
    def plot_feature_contributions(age, education, experience, predicted):
        """
        Plots feature contributions for prediction.
        Args:
        - age (float): Contribution of Age.
        - education (float): Contribution of Education.
        - experience (float): Contribution of Experience.
        - predicted (float): Predicted job level.
        """
        feature_values = {
            "Age Factor": age,
            "Experience Factor": experience,
            "Education Factor": education,
            "Predicted Job Level": predicted
        }

        # Create the bar plot
        features = list(feature_values.keys())
        values = list(feature_values.values())

        plt.figure(figsize=(10, 6))
        sns.barplot(x=features, y=values, palette="viridis")
        plt.title("Feature Contributions to Predicted Job Level")
        plt.ylabel("Contribution Value")
        plt.xlabel("Features")
        plt.xticks(rotation=45)

        # Annotate the bars with their values
        for i, value in enumerate(values):
            plt.text(i, value + 0.05, f"{value:.2f}", ha='center', va='bottom', fontsize=10)

        plt.grid(axis="y", linestyle="--", alpha=0.7)
        st.pyplot(plt)

    # Predict button
    if st.button("Predict Job Level"):
        try:
            # Predict using the loaded model
            prediction = model.predict(scaled_data)

            # Contributions for the visualization
            age_contribution = (age - 20) / 10
            education_contribution = sum([v for k, v in input_data.items() if "Education_" in k])
            experience_contribution = sum([v for k, v in input_data.items() if "ExperienceLevel_" in k])
            predicted_job_level = prediction[0]

            # Normalize contributions
            total_contribution = age_contribution + education_contribution + experience_contribution
            if total_contribution != 0:
                age_contribution /= total_contribution
                education_contribution /= total_contribution
                experience_contribution /= total_contribution

            # Scale contributions relative to the prediction
            age_contribution *= predicted_job_level
            education_contribution *= predicted_job_level
            experience_contribution *= predicted_job_level

            # Display the prediction
            st.success(f"### Predicted Job Level: {round(predicted_job_level, 2)}")

            # Call the visualization function
            plot_feature_contributions(
                age_contribution, education_contribution, experience_contribution, predicted_job_level
            )

        except Exception as e:
            st.error(f"Error making prediction: {e}")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("This app predicts job levels using a trained Support Vector Regression (SVR) model.")
    st.sidebar.markdown("Ensure all required files (`svr_model.pkl` and `scaler.pkl`) are uploaded.")

if __name__ == "__main__":
    app()
