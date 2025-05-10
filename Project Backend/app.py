from flask import Flask, request, jsonify, render_template
import pandas as pd
from model import FraudModel

# Initialize Flask app
app = Flask(__name__)

# Load the fraud detection model
model = FraudModel()

@app.route('/')
def home():
    return render_template('index.html')  # Serves the frontend page

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure that a file is included in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    try:
        # Read the uploaded file (assumed to be a CSV with transaction data)
        df = pd.read_csv(file)
        
        # Make predictions using the fraud model
        preds, probs = model.predict(df)
        
        # Prepare the response with predictions and probabilities
        response = [{"Prediction": int(p), "Probability": float(round(prob * 100, 2))} for p, prob in zip(preds, probs)]
        
        return jsonify(response)  # Return predictions in JSON format
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app in debug mode
if __name__ == '__main__':
    app.run(debug=True)
