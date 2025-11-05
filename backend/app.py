from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import joblib
import io
import os
import logging
import sys

# Configure logging to stderr so it shows in Render logs
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "lightgbm_modelver2.pkl")

# Load model once at startup
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"❌ Failed to load model: {e}")

def prepare_data(df):
    try:
        train_features = model.booster_.feature_name()
    except Exception:
        train_features = getattr(model, "feature_name_", None) or list(df.columns)
    
    present_features = [c for c in train_features if c in df.columns]
    if len(present_features) == 0:
        raise ValueError("No training features found in uploaded file.")
    
    X_new = df[present_features].copy()
    
    for col in X_new.columns:
        if X_new[col].dtype == "object" or str(X_new[col].dtype).startswith("category"):
            X_new[col] = X_new[col].astype(str).astype("category")
    
    X_new = X_new[X_new.notnull().all(axis=1)]
    preds = model.predict(X_new)
    df.loc[X_new.index, "Predicted_EOD_WESM_Price"] = preds
    
    return df, X_new.index

@app.route("/", methods=["GET"])
def home():
    """Root endpoint for health checks"""
    return jsonify({
        "status": "running",
        "service": "WESM Price Prediction API",
        "model_loaded": model is not None
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.route("/predict-file", methods=["POST"])
def predict_file():
    """Returns predictions as an Excel file"""
    try:
        if model is None:
            logger.error("Model is None - not loaded")
            return jsonify({"error": "Model not loaded on server."}), 500
        
        if "file" not in request.files:
            logger.error("No file in request")
            return jsonify({"error": "No file uploaded"}), 400
        
        f = request.files["file"]
        logger.info(f"Received file: {f.filename}")
        
        sheet_name = request.form.get("sheet_name", 0)
        logger.info(f"Reading sheet: {sheet_name}")
        
        df = pd.read_excel(f, sheet_name=sheet_name)
        logger.info(f"DataFrame loaded with shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        df, _ = prepare_data(df)
        logger.info("Data prepared successfully")
        
        output = io.BytesIO()
        df.to_excel(output, index=False)
        output.seek(0)
        
        logger.info("Sending file response")
        return send_file(
            output,
            as_attachment=True,
            download_name="predicted_output.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as e:
        logger.error(f"ERROR in predict_file: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/predict-json", methods=["POST"])
def predict_json():
    """Returns predictions as JSON array"""
    try:
        if model is None:
            return jsonify({"error": "Model not loaded on server."}), 500
        
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        f = request.files["file"]
        sheet_name = request.form.get("sheet_name", 0)
        df = pd.read_excel(f, sheet_name=sheet_name)
        df, valid_idx = prepare_data(df)
        
        preds = df.loc[valid_idx, "Predicted_EOD_WESM_Price"].tolist()
        return jsonify({"predictions": preds, "count": len(preds)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ❌ REMOVE THIS ENTIRE BLOCK FOR RENDER:
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT") or 4000)
#     print(f"🚀 Starting server on 0.0.0.0:{port}")
#     app.run(host='0.0.0.0', port=port, debug=False)

