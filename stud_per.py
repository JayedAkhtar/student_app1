import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the trained model, scaler, and label encoder
def load_model():
    with open("student_lr_final_model.pkl", "rb") as file:
        model, scaler, le = pickle.load(file)
    return model, scaler, le

# Preprocessing the input data
def preprocessing_input(data, scaler, le):
    # Validate 'Extracurricular Activities' key
    if 'Extracurricular Activities' not in data:
        raise KeyError("Missing key: 'Extracurricular Activities' in input data.")
    
    # Handle label encoding for 'Yes'/'No'
    if data['Extracurricular Activities'] not in le.classes_:
        le.classes_ = np.append(le.classes_, data['Extracurricular Activities'])
    
    data['Extracurricular Activities'] = le.transform([data['Extracurricular Activities']])[0]
    
    # Create dataframe for scaler
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

# Predicting the student performance
def predict_data(data):
    model, scaler, le = load_model()
    processed_data = preprocessing_input(data, scaler, le)
    prediction = model.predict(processed_data)
    return prediction

# Main Streamlit App
def main():
    st.title("🎓 Student Performance Prediction")
    st.write("Enter your data to get a prediction for your performance.")

    # Input fields for user data
    hour_studies = st.number_input("📚 Hours Studied", min_value=1, max_value=10, value=5)
    previous_score = st.number_input("📈 Previous Scores (%)", min_value=40, max_value=100, value=70)
    extra = st.selectbox("🎨 Extracurricular Activities", ['Yes', 'No'])
    sleeping_hour = st.number_input("💤 Sleep Hours", min_value=4, max_value=10, value=7)
    number_of_paper_solved = st.number_input("📑 Sample Question Papers Practiced", min_value=0, max_value=10, value=5)

    # Collecting input data in a dictionary
    user_data = {
        "Hours Studied": hour_studies,
        "Previous Scores": previous_score,
        "Extracurricular Activities": extra,  # Corrected spelling
        "Sleep Hours": sleeping_hour,
        "Sample Question Papers Practiced": number_of_paper_solved
    }

    # Button to make predictions
    if st.button("🚀 Predict your score"):
        try:
            prediction = predict_data(user_data)
            st.success(f"✅ Your Predicted Score is: {prediction[0]:.2f}")
        except KeyError as e:
            st.error(f"KeyError: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
