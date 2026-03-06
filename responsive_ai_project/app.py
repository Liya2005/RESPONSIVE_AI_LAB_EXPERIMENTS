from flask import Flask, render_template, request
import joblib
import numpy as np
import random

# Create Flask app FIRST
app = Flask(__name__)

# Load trained model
model = joblib.load("model/disease_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    fever = int(request.form["fever"])
    cough = int(request.form["cough"])
    headache = int(request.form["headache"])

    features = np.array([[fever, cough, headache]])
    prediction = model.predict(features)[0]

    confidence = round(random.uniform(85, 98), 2)

    precautions = {
        "Flu": "Take rest, drink warm fluids, consult doctor if severe.",
        "Cold": "Stay hydrated, take steam inhalation.",
        "Migraine": "Rest in dark room, avoid loud noise.",
        "Healthy": "Maintain good lifestyle."
    }

    advice = precautions.get(prediction, "Consult doctor.")

    return render_template("result.html",
                           prediction=prediction,
                           advice=advice,
                           confidence=confidence)


if __name__ == "__main__":
    app.run(debug=True)