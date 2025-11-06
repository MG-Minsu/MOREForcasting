from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import io
import os
import traceback
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Filter health checks
class HealthCheckFilter(logging.Filter):
    def filter(self, record):
        message = record.getMessage()
        if 'Go-http-client' in message and ('GET /' in message or 'GET /health' in message):
            return False
        return True

werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.addFilter(HealthCheckFilter())

app = Flask(__name__)

# CORS configuration
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Increase max file size (16MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Model path configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), "lightgbm_model_optimized.pkl")

# Global model variable
model = None

def load_model():
    """Load model at startup"""
    global model, MODEL_PATH
    
    try:
        logger.info(f"Attempting to load model from: {MODEL_PATH}")
        
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at {MODEL_PATH}")
            # Try alternate locations
            search_paths = [
                "lightgbm_model_optimized.pkl",
                "./lightgbm_model_optimized.pkl",
                "../lightgbm_model_optimized.pkl",
                "/opt/render/project/src/lightgbm_model_optimized.pkl",
                "/opt/render/project/src/backend/lightgbm_model_optimized.pkl"
            ]
            for path in search_paths:
                if os.path.exists(path):
                    logger.info(f"Found model at: {path}")
                    MODEL_PATH = path
                    break
            else:
                logger.error("Model file not found in any location")
                return False
        
        model = joblib.load(MODEL_PATH)
        logger.info(f"✅ Model loaded successfully")
        
        # Get features
        features = model.booster_.feature_name()
        logger.info(f"Model has {len(features)} features")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# Load model on startup
logger.info("="*60)
logger.info("STARTING WESM PRICE PREDICTION API v4.0")
logger.info("="*60)
model_loaded = load_model()
logger.info(f"Model loaded: {model_loaded}")
logger.info("="*60)

def prepare_data_for_prediction(new_data):
    """
    Prepare data for prediction following notebook logic
    """
    try:
        if model is None:
            raise ValueError("Model is not loaded")
        
        logger.info(f"Processing {len(new_data)} rows with {len(new_data.columns)} columns")
        
        # Get training features
        train_features = model.booster_.feature_name()
        logger.info(f"Model expects {len(train_features)} features")
        
        # Check for missing features
        missing_features = [f for f in train_features if f not in new_data.columns]
        present_features = [f for f in train_features if f in new_data.columns]
        
        logger.info(f"Present: {len(present_features)}, Missing: {len(missing_features)}")
        
        # Fill missing features with 0
        if missing_features:
            logger.info(f"Filling {len(missing_features)} missing features with 0")
            for feature in missing_features:
                new_data[feature] = 0
        
        # Select features in correct order
        X_new = new_data[train_features].copy()
        
        # Convert categorical columns
        object_cols = X_new.select_dtypes(include=["object"]).columns.tolist()
        if object_cols:
            logger.info(f"Converting {len(object_cols)} object columns to category")
            for col in object_cols:
                X_new[col] = X_new[col].astype("category")
        
        # Get valid rows (no NaN)
        valid_mask = X_new.notnull().all(axis=1)
        X_new_clean = X_new[valid_mask]
        valid_indices = X_new_clean.index.tolist()
        
        logger.info(f"Valid rows: {len(valid_indices)} out of {len(new_data)}")
        
        if len(valid_indices) == 0:
            return new_data, np.array([]), []
        
        # Make predictions
        logger.info("Making predictions...")
        predicted_prices = model.predict(X_new_clean)
        
        logger.info(f"✅ Predictions: Min={predicted_prices.min():.2f}, Max={predicted_prices.max():.2f}, Mean={predicted_prices.mean():.2f}")
        
        # Add predictions to dataframe
        result_df = new_data.copy()
        result_df["Predicted_EOD_WESM_Price"] = np.nan
        result_df.loc[valid_indices, "Predicted_EOD_WESM_Price"] = predicted_prices
        
        return result_df, predicted_prices, valid_indices
        
    except Exception as e:
        logger.error(f"Error in prepare_data_for_prediction: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.route("/", methods=["GET"])
def home():
    """Root endpoint"""
    return jsonify({
        "service": "WESM Price Prediction API",
        "version": "4.0",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "/health": "GET - Health check",
            "/features": "GET - List required features",
            "/predict-file": "POST - Upload Excel, get Excel with predictions",
            "/predict-json": "POST - Upload Excel, get JSON with table data",
            "/predict-table": "POST - Upload Excel, get table-ready JSON"
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
            train_features = model.booster_.feature_name()
            status["feature_count"] = len(train_features)
            status["sample_features"] = train_features[:5]
        except Exception as e:
            status["feature_error"] = str(e)
    
    return jsonify(status), 200 if model is not None else 503

@app.route("/features", methods=["GET"])
def list_features():
    """List all required features"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        train_features = model.booster_.feature_name()
        return jsonify({
            "total_features": len(train_features),
            "features": train_features
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict-file", methods=["POST", "OPTIONS"])
def predict_file():
    """Upload Excel and get Excel back with predictions"""
    
    if request.method == "OPTIONS":
        return "", 200
    
    try:
        logger.info("="*60)
        logger.info("PREDICT-FILE REQUEST")
        logger.info("="*60)
        
        if model is None:
            return jsonify({"error": "Model not loaded"}), 503
        
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files["file"]
        if not file or file.filename == "":
            return jsonify({"error": "Empty filename"}), 400
        
        logger.info(f"File: {file.filename}")
        
        # Get sheet name
        sheet_name = request.form.get("sheet_name", "0")
        try:
            sheet_name = int(sheet_name) if sheet_name.isdigit() else sheet_name
        except:
            sheet_name = 0
        
        # Read Excel
        file_content = file.read()
        new_data = pd.read_excel(io.BytesIO(file_content), sheet_name=sheet_name)
        
        if new_data.empty:
            return jsonify({"error": "Empty file"}), 400
        
        # Predict
        result_df, predictions, valid_indices = prepare_data_for_prediction(new_data)
        
        if len(predictions) == 0:
            return jsonify({"error": "No valid rows to predict"}), 400
        
        # Create Excel output
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            result_df.to_excel(writer, index=False, sheet_name='Predictions')
        output.seek(0)
        
        logger.info(f"✅ Returning Excel with {len(predictions)} predictions")
        
        return send_file(
            output,
            as_attachment=True,
            download_name="predicted_output.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/predict-json", methods=["POST", "OPTIONS"])
@app.route("/predict-table", methods=["POST", "OPTIONS"])
def predict_json():
    """Upload Excel and get table data as JSON"""
    
    if request.method == "OPTIONS":
        return "", 200
    
    try:
        logger.info("="*60)
        logger.info("PREDICT-JSON/TABLE REQUEST")
        logger.info("="*60)
        
        if model is None:
            return jsonify({"error": "Model not loaded"}), 503
        
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files["file"]
        if not file or file.filename == "":
            return jsonify({"error": "Empty filename"}), 400
        
        logger.info(f"File: {file.filename}")
        
        # Get sheet name
        sheet_name = request.form.get("sheet_name", "0")
        try:
            sheet_name = int(sheet_name) if sheet_name.isdigit() else sheet_name
        except:
            sheet_name = 0
        
        # Read Excel
        file_content = file.read()
        new_data = pd.read_excel(io.BytesIO(file_content), sheet_name=sheet_name)
        
        if new_data.empty:
            return jsonify({"error": "Empty file"}), 400
        
        logger.info(f"Read {len(new_data)} rows, {len(new_data.columns)} columns")
        
        # Predict
        result_df, predictions, valid_indices = prepare_data_for_prediction(new_data)
        
        if len(predictions) == 0:
            return jsonify({
                "success": False,
                "error": "No valid rows to predict",
                "total_rows": len(new_data)
            }), 400
        
        # Convert result to table format
        # Replace NaN with null for JSON
        result_df = result_df.replace({np.nan: None})
        
        # Convert to records (array of objects)
        table_data = result_df.to_dict('records')
        
        # Get column names
        columns = list(result_df.columns)
        
        # Get feature info
        train_features = model.booster_.feature_name()
        present = [f for f in train_features if f in new_data.columns]
        missing = [f for f in train_features if f not in new_data.columns]
        
        logger.info(f"✅ Returning {len(table_data)} rows of table data")
        
        response = {
            "success": True,
            "data": table_data,  # Array of row objects for table
            "columns": columns,  # Column names
            "total_rows": len(new_data),
            "predicted_rows": len(valid_indices),
            "skipped_rows": len(new_data) - len(valid_indices),
            "prediction_stats": {
                "min": float(predictions.min()),
                "max": float(predictions.max()),
                "mean": float(predictions.mean()),
                "median": float(np.median(predictions))
            },
            "feature_info": {
                "required": len(train_features),
                "provided": len(present),
                "missing": len(missing),
                "missing_names": missing[:10]  # First 10 for brevity
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e),
            "type": type(e).__name__
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
