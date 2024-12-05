import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained SVR model
with open('svr_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define the full list of trained columns
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

# Streamlit App
def app():
    st.set_page_config(page_title="Job Level Predictor", layout="wide", theme="dark")
    st.title("Job Level Predictor (SVR)")

    # Model and Scaler Load Status
    st.sidebar.success("✔️ SVR Model loaded successfully.")
    st.sidebar.success("✔️ Scaler loaded successfully.")

    # Sidebar Inputs
    st.sidebar.header("Input Features")
    st.sidebar.text("Enter the features below:")

    # User Inputs
    age = st.sidebar.slider("Age", 18, 75, 30)
    experience_level = st.sidebar.selectbox(
        "Experience Level", [
            "Director",
            "Entry level",
            "Executive",
            "Internship",
            "Mid-Senior level"
        ]
    )
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

    # Create input DataFrame
    input_data = pd.DataFrame(columns=TRAINED_COLUMNS)

    # Fill with zeros initially
    input_data.loc[0] = [0] * len(TRAINED_COLUMNS)

    # Map user inputs to DataFrame
    input_data['Age'] = age
    input_data[f'ExperienceLevel_{experience_level}'] = 1
    input_data[f'Education_{education_level}'] = 1

    # Display input data
    st.write("### Input Data")
    st.write(input_data)

    # Scale the input data
    try:
        scaled_data = scaler.transform(input_data)
        st.write("### Scaled Input Data")
        st.write(scaled_data)

        # Predict button
        if st.button("Predict Job Level"):
            prediction = model.predict(scaled_data)
            st.success(f"### Predicted Job Level: {round(prediction[0], 2)}")

    except Exception as e:
        st.error(f"Error scaling input data: {e}")

if __name__ == "__main__":
    app()
