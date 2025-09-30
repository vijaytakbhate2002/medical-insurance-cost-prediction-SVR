from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import logging
import pandas as pd

app = Flask(__name__)

model = joblib.load("best_performing_model/InsuranceCostPredictor_13.pkl")
scaler = joblib.load("data/target_scaling.pkl")


@app.route("/")
def index():
    return render_template("index.html")  


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json() 

        age = int(data.get("age", 0))
        sex = data.get("sex", "male").lower()
        bmi = float(data.get("bmi", 0))

        children = int(data.get("children", 0))
        smoker = data.get("smoker", "no").lower()
        region = data.get("(US) region", "southwest").lower()
        
        input_data = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region
        }])

        prediction = model.predict(input_data)
        print("before:", prediction)
        prediction = scaler.inverse_transform(prediction.reshape(-1, 1))
        print("after:", prediction)

        return jsonify({"prediction": float(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
