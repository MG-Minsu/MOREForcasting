from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import joblib
import io
import os
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# CORS configuration for production
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Model path configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), "LIGHTGBM-1.pkl")

# Global model variable
model = None

def load_model():
    """Load model at startup"""
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at {MODEL_PATH}")
            return False
        
        model = joblib.load(MODEL_PATH)
        logger.info(f"✅ Model loaded successfully from {MODEL_PATH}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# Load model on startup
load_model()

def get_model_features():
    """Extract feature names from the model"""
    try:
        if hasattr(model, 'feature_name_'):
            return model.feature_name_
        elif hasattr(model, 'booster_'):
            return model.booster_.feature_name()
        elif hasattr(model, 'feature_names_in_'):
            return list(model.feature_names_in_)
        else:
            logger.warning("Could not extract feature names from model")
            return None
    except Exception as e:
        logger.error(f"Error getting model features: {str(e)}")
        return None

def prepare_data(df):
    """
    Prepares input dataframe for prediction
    Returns: (result_df, predictions, valid_indices)
    """
    if model is None:
        raise ValueError("Model is not loaded")
    
    # Get expected features from model
    expected_features = get_model_features()
    
    if expected_features is None:
        # Fallback: use all columns in dataframe
        logger.warning("Using all dataframe columns as features")
        expected_features = list(df.columns)
    
    # Find which features are present
    present_features = [col for col in expected_features if col in df.columns]
    missing_features = [col for col in expected_features if col not in df.columns]
    
    if len(present_features) == 0:
        raise ValueError(
            f"No training features found in uploaded file.\n"
            f"Expected features: {expected_features[:10]}\n"
            f"Found columns: {list(df.columns)[:10]}"
        )
    
    if missing_features:
        logger.warning(f"{len(missing_features)} features missing from input data")
    
    # Create prediction dataset
    X_pred = df[present_features].copy()
    
    # Handle categorical columns
    for col in X_pred.columns:
        if X_pred[col].dtype == "object":
            X_pred[col] = X_pred[col].astype(str).astype("category")
        elif pd.api.types.is_categorical_dtype(X_pred[col]):
            X_pred[col] = X_pred[col].astype("category")
    
    # Find valid rows (no missing values)
    valid_mask = X_pred.notnull().all(axis=1)
    valid_indices = X_pred[valid_mask].index.tolist()
    
    if len(valid_indices) == 0:
        raise ValueError("All rows contain missing values in required features")
    
    # Get valid data
    X_valid = X_pred.loc[valid_indices]
    
    # Make predictions
    try:
        predictions = model.predict(X_valid)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise ValueError(f"Model prediction failed: {str(e)}")
    
    # Create result dataframe
    result_df = df.copy()
    result_df["Predicted_EOD_WESM_Price"] = None
    result_df.loc[valid_indices, "Predicted_EOD_WESM_Price"] = predictions
    
    return result_df, predictions, valid_indices

@app.route("/", methods=["GET"])
def home():
    """Root endpoint"""
    return jsonify({
        "service": "WESM Price Prediction API",
        "status": "running",
        "endpoints": {
            "/health": "GET - Health check",
            "/predict-file": "POST - Upload Excel, get Excel with predictions",
            "/predict-json": "POST - Upload Excel, get JSON predictions"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    status = {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH)
    }
    
    if model is not None:
        try:
            features = get_model_features()
            if features:
                status["feature_count"] = len(features)
                status["sample_features"] = features[:5]
        except Exception as e:
            status["feature_error"] = str(e)
    
    return jsonify(status), 200 if model is not None else 503

@app.route("/predict-file", methods=["POST"])
def predict_file():
    """Upload Excel and get Excel back with predictions"""
    try:
        # Check model
        if model is None:
            return jsonify({"error": "Model not loaded on server"}), 503
        
        # Check file upload
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files["file"]
        
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400
        
        # Get sheet name
        sheet_name = request.form.get("sheet_name", "0")
        try:
            sheet_name = int(sheet_name) if sheet_name.isdigit() else sheet_name
        except:
            sheet_name = 0
        
        # Read Excel file
        try:
            df = pd.read_excel(file, sheet_name=sheet_name)
        except Exception as e:
            return jsonify({"error": f"Failed to read Excel file: {str(e)}"}), 400
        
        if df.empty:
            return jsonify({"error": "Uploaded file is empty"}), 400
        
        logger.info(f"Processing file with {len(df)} rows and {len(df.columns)} columns")
        
        # Make predictions
        result_df, predictions, valid_indices = prepare_data(df)
        
        # Create Excel output
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            result_df.to_excel(writer, index=False, sheet_name='Predictions')
        output.seek(0)
        
        logger.info(f"✅ Successfully predicted {len(predictions)} rows")
        
        return send_file(
            output,
            as_attachment=True,
            download_name="predicted_output.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    except Exception as e:
        logger.error(f"Error in predict-file: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "type": type(e).__name__
        }), 500

@app.route("/predict-json", methods=["POST"])
def predict_json():
    """Upload Excel and get predictions as JSON"""
    try:
        # Check model
        if model is None:
            return jsonify({"error": "Model not loaded on server"}), 503
        
        # Check file upload
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files["file"]
        
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400
        
        # Get sheet name
        sheet_name = request.form.get("sheet_name", "0")
        try:
            sheet_name = int(sheet_name) if sheet_name.isdigit() else sheet_name
        except:
            sheet_name = 0
        
        # Read Excel file
        try:
            df = pd.read_excel(file, sheet_name=sheet_name)
        except Exception as e:
            return jsonify({"error": f"Failed to read Excel file: {str(e)}"}), 400
        
        if df.empty:
            return jsonify({"error": "Uploaded file is empty"}), 400
        
        logger.info(f"Processing file with {len(df)} rows and {len(df.columns)} columns")
        
        # Make predictions
        result_df, predictions, valid_indices = prepare_data(df)
        
        logger.info(f"✅ Successfully predicted {len(predictions)} rows")
        
        # Return predictions with metadata
        return jsonify({
            "success": True,
            "predictions": predictions.tolist(),
            "total_rows": len(df),
            "predicted_rows": len(valid_indices),
            "skipped_rows": len(df) - len(valid_indices),
            "valid_indices": valid_indices
        })
    
    except Exception as e:
        logger.error(f"Error in predict-json: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e),
            "type": type(e).__name__
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
