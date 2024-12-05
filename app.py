import streamlit as st
import pickle
import pandas as pd
import numpy as np

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

    # Default output
    st.write("👈 Adjust the inputs in the sidebar and click **Predict Job Level** to see the results!")

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
