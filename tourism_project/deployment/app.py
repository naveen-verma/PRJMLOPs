import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="nv185001/churn-model", filename="best_tourism_prediction_model.joblib")
# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Tourist Prediction App")
st.write("The Tourist Prediction App is an internal tool for staff to predicts whether the customer would pruchase the tourist package.")
st.write("Kindly enter the tourist details to check whether they are likely to purchase the plan")

Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=100, value=30)
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.number_input("Duration Of Pitch (duration of the sales pitch delivered to the customer", min_value=1, max_value=100, value=30)
NumberOfPersonVisiting = st.number_input("Number Of Person Visiting ", min_value=1)
NumberOfFollowups = st.number_input("Number Of Followups ", min_value=1)
PreferredPropertyStar = st.number_input("Preferred Property Star ", min_value=1, max_value=5)
NumberOfTrips = st.number_input("Number Of Trips ", min_value=1)
Passport = st.selectbox("Passport", [0, 1])
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score ", min_value=1, max_value=5)
OwnCar = st.selectbox("Own Car", [0, 1])
NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting ", min_value=0)
MonthlyIncome = st.number_input("Monthly Income ", min_value=0)

TypeofContact = st.selectbox("Type Of Contact", ["Company Invited", "Self Inquiry"])
Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
Designation = st.selectbox("Designation", ["AVP", "Executive", "Manager", "Senior Manager", "VP"])

# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'ProductPitched': ProductPitched,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "to buy the package" if prediction == 1 else "that will not buy the package"
    st.write(f"Based on the information provided, the customer is likely {result}.")

