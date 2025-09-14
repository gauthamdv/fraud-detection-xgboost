from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

MODEL_PATH = "Models/xgb_fraud_model.pkl"
model = None
feature_names = None


def load_model():
    global model, feature_names
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    feature_names = model.get_booster().feature_names
    print("Model loaded successfully.")


def prepare_input(input_data):
    df = pd.DataFrame([input_data])
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]
    return df


@app.route("/", methods=["GET", "POST"])
def home():
    result_text = ""
    if request.method == "POST":
        data = {
            "amount": float(request.form["amount"]),
            "oldbalanceOrg": float(request.form["oldbalanceOrg"]),
            "newbalanceOrig": float(request.form["newbalanceOrig"]),
            "oldbalanceDest": float(request.form["oldbalanceDest"]),
            "newbalanceDest": float(request.form["newbalanceDest"]),
            "type_TRANSFER": 1 if request.form["type"] == "TRANSFER" else 0,
            "type_CASH_OUT": 1 if request.form["type"] == "CASH_OUT" else 0,
            "type_PAYMENT": 1 if request.form["type"] == "PAYMENT" else 0,
            "type_DEBIT": 1 if request.form["type"] == "DEBIT" else 0,
            "isFlaggedFraud": 0
        }

        input_df = prepare_input(data)
        prediction_proba = model.predict_proba(input_df)[:, 1][0]
        prediction_class = model.predict(input_df)[0]
        
        if prediction_class == 1:
            label = "Fraud"
            confidence = prediction_proba
        else:
            label = "Not Fraud"
            confidence = 1 - prediction_proba

        result_text = f"Results\nPrediction: {label}\nConfidence: {round(float(confidence),7)}"
    
        


    return render_template("index.html", result_text=result_text)



if __name__ == "__main__":
    load_model()
    app.run(debug=True)
