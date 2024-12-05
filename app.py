import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained SVR model
try:
    with open('svr_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("The SVR model file (svr_model.pkl) is missing. Please upload it.")
    st.stop()

# Load the scaler
try:
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    st.error("The scaler file (scaler.pkl) is missing. Please upload it.")
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
        "Bachelor’s degree": 1,
        "Master’s degree": 2,
        "Doctoral degree": 3
    }
    experience_mapping = {
        "Entry level": 1,
        "Mid-Senior level": 2,
        "Executive": 3
    }

    # Map inputs to numeric values
    education_numeric = education_mapping[education_level]
    experience_numeric = experience_mapping[experience_level]

    # Create input DataFrame
    input_data = pd.DataFrame({
        'Age': [age],
        'Education': [education_numeric],
        'ExperienceLevel': [experience_numeric],
    })

    # Display raw input data
    st.write("### Input Data")
    st.write(input_data)

    # Scale the input data
    try:
        scaled_data = scaler.transform(input_data)
    except Exception as e:
        st.error(f"Error scaling input data: {e}")
        st.stop()

    # Display scaled input data
    st.write("### Input Data (Scaled)")
    st.write(scaled_data)

    # Predict button
    if st.button("Predict Job Level"):
        try:
            # Predict using the loaded model
            prediction = model.predict(scaled_data)
            
            # Map prediction back to job level labels
            job_level_mapping = {
                1: "Entry-Level",
                2: "Mid-Level",
                3: "Senior-Level",
                4: "Executive-Level",
                5: "C-Level"
            }
            predicted_label = job_level_mapping.get(round(prediction[0]), "Unknown")

            # Display the prediction
            st.success(f"### Predicted Job Level: {predicted_label}")
            st.write(f"Numerical Prediction: {round(prediction[0], 2)}")

        except Exception as e:
            st.error(f"Error making prediction: {e}")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("This app predicts job levels using a trained Support Vector Regression (SVR) model.")
    st.sidebar.markdown("Ensure all required files (`svr_model.pkl` and `scaler.pkl`) are uploaded.")

if __name__ == "__main__":
    app()
