from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import joblib
import io
import os
app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "LIGHTGBM-1.pkl")

# Load model once at startup
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"❌ Failed to load model: {e}")


def prepare_data(df):
    """Prepares input dataframe for prediction"""
    try:
        train_features = model.booster_.feature_name()
    except Exception:
        train_features = getattr(model, "feature_name_", None) or list(df.columns)

    present_features = [c for c in train_features if c in df.columns]
    if len(present_features) == 0:
        raise ValueError("No training features found in uploaded file.")

    X_new = df[present_features].copy()

    # Convert categorical
    for col in X_new.columns:
        if X_new[col].dtype == "object" or str(X_new[col].dtype).startswith("category"):
            X_new[col] = X_new[col].astype(str).astype("category")

    # Drop NaN rows
    X_new = X_new[X_new.notnull().all(axis=1)]

    preds = model.predict(X_new)
    df.loc[X_new.index, "Predicted_EOD_WESM_Price"] = preds

    return df, X_new.index

@app.route('/')
def home():
    return jsonify({"message": "Backend running OK"})
@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route('/predict-file', methods=['POST'])
def predict_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        df = pd.read_csv(file)

        print("✅ Uploaded file columns:", df.columns.tolist(), flush=True)

        # Run prediction
        preds = model.predict(df)

        print("✅ Prediction completed. Sample output:", preds[:5], flush=True)

        output = io.BytesIO()
        df['Prediction'] = preds
        df.to_csv(output, index=False)
        output.seek(0)

        return send_file(output, as_attachment=True, download_name='predictions.csv', mimetype='text/csv')

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/predict-json", methods=["POST"])
def predict_json():
    """Upload Excel and get predictions as JSON array"""
    try:
        if model is None:
            return jsonify({"error": "Model not loaded on server."}), 500

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        f = request.files["file"]

        # Get sheet_name from form, default to 0 if not provided
        sheet_name = request.form.get("sheet_name", 0)

        df = pd.read_excel(f, sheet_name=sheet_name)
        df, valid_idx = prepare_data(df)

        preds = df.loc[valid_idx, "Predicted_EOD_WESM_Price"].tolist()
        return jsonify(preds)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


