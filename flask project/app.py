import numpy as np
from flask import Flask, request, render_template
import pickle

Flask_app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@Flask_app.route("/")
def Home():
    return render_template("index.html")

@Flask_app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]

    prediction = model.predict(features)
    probability = model.predict_proba(features)

    confidence = round(max(probability[0]) * 100, 2)

    return render_template(
        "index.html",
        prediction_text=f"The Predicted Crop is {prediction[0]}",
        confidence_text=f"Confidence: {confidence}%"
    )

if __name__ == "__main__":
    Flask_app.run(debug=True)