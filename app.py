import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({"predicted_count": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
