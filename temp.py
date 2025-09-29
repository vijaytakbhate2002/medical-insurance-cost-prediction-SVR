import joblib
import pandas as pd

# Load model
model = joblib.load("best_performing_model/InsuranceCostPredictor_13.pkl")
scaler = joblib.load("data/target_scaling.pkl")

# Create DataFrame with correct column names
input_data = pd.DataFrame([{
    "age": 30,
    "sex": "male",
    "bmi": 25.0,
    "children": 0,
    "smoker": "no",
    "region": "southeast"
}])

# Predict
prediction = model.predict(input_data)

print("before:", prediction)
prediction = target_scaler.inverse_transform(prediction.reshape(-1, 1))
print("after:", prediction)


