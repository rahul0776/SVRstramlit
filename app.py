import streamlit as st
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

# Set page configuration
st.set_page_config(page_title="Job Level Predictor", layout="wide")

# Load the trained SVR model
with open('svr_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define trained columns (features used during training)
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

# Function to plot feature contributions
def plot_feature_contributions(scaled_data, feature_names):
    """
    Visualize the feature contributions using a bar plot.
    :param scaled_data: Scaled input data (array).
    :param feature_names: List of feature names.
    """
    contribution_df = pd.DataFrame(scaled_data, columns=feature_names)
    contribution_df = contribution_df.T  # Transpose for easier plotting
    contribution_df.columns = ["Scaled Value"]

    # Plot the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=contribution_df.index,
        y=contribution_df["Scaled Value"],
        palette="viridis",
        ax=ax
    )
    plt.title("Feature Contributions to Predicted Job Level", fontsize=16)
    plt.ylabel("Contribution Value", fontsize=12)
    plt.xlabel("Features", fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()

    # Return the plot as a buffer
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

# Streamlit App
def app():
    st.title("Job Level Predictor (SVR)")

    # Model and Scaler Load Status
    st.sidebar.success("SVR Model loaded successfully.")
    st.sidebar.success("Scaler loaded successfully.")

    # Sidebar Inputs
    st.sidebar.header("Enter the features below:")
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

    # One-hot encode education and experience level
    education_mapping = {
        "Doctoral degree": "Education_Doctoral degree",
        "I prefer not to answer": "Education_I prefer not to answer",
        "Master’s degree": "Education_Master’s degree",
        "No formal education past high school": "Education_No formal education past high school",
        "Professional degree": "Education_Professional degree",
        "Some college/university study without earning a bachelor’s degree": "Education_Some college/university study without earning a bachelor’s degree"
    }
    experience_mapping = {
        "Director": "ExperienceLevel_Director",
        "Entry level": "ExperienceLevel_Entry level",
        "Executive": "ExperienceLevel_Executive",
        "Internship": "ExperienceLevel_Internship",
        "Mid-Senior level": "ExperienceLevel_Mid-Senior level"
    }

    # Create input DataFrame
    input_data = pd.DataFrame(0, index=[0], columns=TRAINED_COLUMNS)
    input_data['Age'] = age
    input_data[education_mapping[education_level]] = 1
    input_data[experience_mapping[experience_level]] = 1

    # Display Input Data
    st.subheader("Input Data")
    st.write(input_data)

    # Scale the input data
    try:
        scaled_data = scaler.transform(input_data)
        st.subheader("Scaled Data")
        st.write(pd.DataFrame(scaled_data, columns=TRAINED_COLUMNS))
    except Exception as e:
        st.error(f"Error scaling input data: {e}")
        return

    # Predict button
    if st.button("Predict Job Level"):
        try:
            # Predict using the loaded model
            prediction = model.predict(scaled_data)
            st.subheader("Predicted Job Level:")
            st.write(f"### {round(prediction[0], 2)}")

            # Plot Feature Contributions
            st.subheader("Feature Contributions Visualization")
            feature_contribution_plot = plot_feature_contributions(
                scaled_data, TRAINED_COLUMNS
            )
            st.image(feature_contribution_plot, use_column_width=True)

        except Exception as e:
            st.error(f"Error making prediction: {e}")

if __name__ == "__main__":
    app()
