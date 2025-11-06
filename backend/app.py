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

# Filter out Render health checks
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
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"❌ Model file not found at {MODEL_PATH}")
            return False
        
        logger.info(f"Loading model from {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        logger.info(f"✅ Model loaded successfully")
        logger.info(f"Model type: {type(model)}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# Load model on startup
logger.info("=" * 60)
logger.info("Starting WESM Price Prediction API v3.0")
logger.info("Following exact prediction logic from training notebook")
logger.info("=" * 60)
load_model()

def prepare_data_for_prediction(new_data):
    """
    Prepare data for prediction following the exact logic from training notebook
    
    Parameters:
    -----------
    new_data : DataFrame
        Input data from uploaded Excel file
    
    Returns:
    --------
    (result_df, predictions, valid_indices)
    """
    try:
        if model is None:
            raise ValueError("Model is not loaded")
        
        logger.info(f"📊 Input data: {len(new_data)} rows, {len(new_data.columns)} columns")
        logger.info(f"Input columns: {list(new_data.columns)}")
        
        # --- Load feature names used during training ---
        train_features = model.booster_.feature_name()
        logger.info(f"🎯 Model expects {len(train_features)} features")
        logger.info(f"Required features: {train_features}")
        
        # --- Check which features are missing and handle them ---
        missing_features = [f for f in train_features if f not in new_data.columns]
        
        if missing_features:
            logger.warning(f"⚠️ Missing {len(missing_features)} features in testing data:")
            logger.warning(f"Missing features: {missing_features}")
            logger.info("🔧 These features will be filled with 0")
            
            # Fill missing features with 0
            for feature in missing_features:
                new_data[feature] = 0
        else:
            logger.info("✅ All required features present in input data")
        
        # --- Keep only columns used in training ---
        X_new = new_data[train_features].copy()
        logger.info(f"📋 Selected features: {X_new.shape}")
        
        # --- Align categorical columns ---
        # First, identify object columns and convert to category
        object_cols = X_new.select_dtypes(include=["object"]).columns.tolist()
        if object_cols:
            logger.info(f"🔤 Converting {len(object_cols)} object columns to category")
            for col in object_cols:
                X_new[col] = X_new[col].astype("category")
        
        # Log categorical columns
        categorical_cols = X_new.select_dtypes(include="category").columns.tolist()
        if categorical_cols:
            logger.info(f"📊 Categorical columns: {len(categorical_cols)}")
        
        # --- Drop rows with NaN in features ---
        # Check which rows have any NaN values
        valid_mask = X_new.notnull().all(axis=1)
        X_new_clean = X_new[valid_mask]
        valid_indices = X_new_clean.index.tolist()
        
        logger.info(f"✅ Valid rows: {len(valid_indices)} out of {len(new_data)}")
        
        if len(valid_indices) == 0:
            logger.warning("⚠️ No valid rows to predict. All rows have missing values.")
            return new_data, np.array([]), []
        
        # --- Generate predictions ---
        logger.info(f"🔮 Generating predictions for {len(X_new_clean)} rows...")
        logger.info(f"Input shape for prediction: {X_new_clean.shape}")
        
        predicted_prices = model.predict(X_new_clean)
        
        logger.info(f"✅ Predictions successful: {len(predicted_prices)} values")
        logger.info(f"📈 Prediction stats:")
        logger.info(f"   Min: {predicted_prices.min():.2f}")
        logger.info(f"   Max: {predicted_prices.max():.2f}")
        logger.info(f"   Mean: {predicted_prices.mean():.2f}")
        logger.info(f"   Median: {np.median(predicted_prices):.2f}")
        
        # --- Add predictions to dataframe ---
        result_df = new_data.copy()
        result_df["Predicted_EOD_WESM_Price"] = np.nan
        result_df.loc[valid_indices, "Predicted_EOD_WESM_Price"] = predicted_prices
        
        # Add metadata
        result_df["_prediction_status"] = "skipped_missing_values"
        result_df.loc[valid_indices, "_prediction_status"] = "predicted"
        
        logger.info("✅ Predictions added to dataframe")
        
        return result_df, predicted_prices, valid_indices
        
    except Exception as e:
        logger.error(f"❌ Error in prepare_data_for_prediction: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.route("/", methods=["GET"])
def home():
    """Root endpoint"""
    return jsonify({
        "service": "WESM Price Prediction API",
        "version": "3.0",
        "status": "running",
        "model_loaded": model is not None,
        "features": {
            "follows_training_logic": True,
            "auto_fill_missing_features": True,
            "handles_categorical_columns": True,
            "description": "Predictions follow exact logic from training notebook"
        },
        "endpoints": {
            "/health": "GET - Health check",
            "/features": "GET - List required features",
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
            train_features = model.booster_.feature_name()
            status["feature_count"] = len(train_features)
            status["sample_features"] = train_features[:10] if len(train_features) > 10 else train_features
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
            "features": train_features,
            "note": "Missing features will be automatically filled with 0"
        })
    except Exception as e:
        logger.error(f"Error getting features: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict-file", methods=["POST", "OPTIONS"])
def predict_file():
    """Upload Excel and get Excel back with predictions"""
    
    if request.method == "OPTIONS":
        return "", 200
    
    try:
        logger.info("\n" + "=" * 60)
        logger.info("PREDICT-FILE REQUEST")
        logger.info("=" * 60)
        
        # Check model
        if model is None:
            return jsonify({
                "error": "Model not loaded on server",
                "details": "The model file is missing or failed to load"
            }), 503
        
        # Check file upload
        if "file" not in request.files:
            return jsonify({
                "error": "No file uploaded",
                "details": "Please include a file with the key 'file'"
            }), 400
        
        file = request.files["file"]
        
        if not file or file.filename == "":
            return jsonify({"error": "Empty filename"}), 400
        
        logger.info(f"📁 Received file: {file.filename}")
        
        # Get sheet name
        sheet_name = request.form.get("sheet_name", "0")
        try:
            sheet_name = int(sheet_name) if sheet_name.isdigit() else sheet_name
        except:
            sheet_name = 0
        
        logger.info(f"📊 Using sheet: {sheet_name}")
        
        # Read Excel file
        try:
            file_content = file.read()
            logger.info(f"📦 File size: {len(file_content):,} bytes")
            
            new_data = pd.read_excel(io.BytesIO(file_content), sheet_name=sheet_name)
            logger.info(f"✅ Excel parsed successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to read Excel: {str(e)}")
            return jsonify({
                "error": "Failed to read Excel file",
                "details": str(e),
                "hint": "Make sure the file is a valid Excel file (.xlsx or .xls)"
            }), 400
        
        if new_data.empty:
            return jsonify({
                "error": "Uploaded file is empty",
                "details": "The Excel file contains no data"
            }), 400
        
        # Make predictions using exact logic from notebook
        result_df, predictions, valid_indices = prepare_data_for_prediction(new_data)
        
        if len(predictions) == 0:
            return jsonify({
                "error": "No valid rows to predict",
                "details": "All rows have missing values in required features"
            }), 400
        
        # Create Excel output
        logger.info("📝 Creating output Excel file...")
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            result_df.to_excel(writer, index=False, sheet_name='Predictions')
        output.seek(0)
        
        logger.info(f"✅ Success! Predicted {len(predictions)} rows")
        logger.info("=" * 60 + "\n")
        
        return send_file(
            output,
            as_attachment=True,
            download_name="predicted_output.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    except Exception as e:
        logger.error(f"\n❌ ERROR in predict-file:")
        logger.error(f"Error: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        logger.error("=" * 60 + "\n")
        
        return jsonify({
            "error": str(e),
            "type": type(e).__name__
        }), 500

@app.route("/predict-json", methods=["POST", "OPTIONS"])
def predict_json():
    """Upload Excel and get predictions as JSON"""
    
    if request.method == "OPTIONS":
        return "", 200
    
    try:
        logger.info("\n" + "=" * 60)
        logger.info("PREDICT-JSON REQUEST")
        logger.info("=" * 60)
        
        # Check model
        if model is None:
            return jsonify({"error": "Model not loaded"}), 503
        
        # Check file upload
        if "file" not in request.files:
            return jsonify({
                "error": "No file uploaded",
                "details": "Please include a file with the key 'file'"
            }), 400
        
        file = request.files["file"]
        
        if not file or file.filename == "":
            return jsonify({"error": "Empty filename"}), 400
        
        logger.info(f"📁 Received file: {file.filename}")
        
        # Get sheet name
        sheet_name = request.form.get("sheet_name", "0")
        try:
            sheet_name = int(sheet_name) if sheet_name.isdigit() else sheet_name
        except:
            sheet_name = 0
        
        # Read Excel file
        try:
            file_content = file.read()
            new_data = pd.read_excel(io.BytesIO(file_content), sheet_name=sheet_name)
            logger.info(f"✅ Excel parsed successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to read Excel: {str(e)}")
            return jsonify({
                "error": "Failed to read Excel file",
                "details": str(e)
            }), 400
        
        if new_data.empty:
            return jsonify({"error": "Uploaded file is empty"}), 400
        
        # Make predictions using exact logic from notebook
        result_df, predictions, valid_indices = prepare_data_for_prediction(new_data)
        
        if len(predictions) == 0:
            return jsonify({
                "success": False,
                "error": "No valid rows to predict",
                "details": "All rows have missing values in required features",
                "total_rows": len(new_data),
                "predicted_rows": 0
            }), 400
        
        # Get feature info
        train_features = model.booster_.feature_name()
        present_features = [f for f in train_features if f in new_data.columns]
        missing_features = [f for f in train_features if f not in new_data.columns]
        
        logger.info(f"✅ Success! Predicted {len(predictions)} rows")
        logger.info("=" * 60 + "\n")
        
        return jsonify({
            "success": True,
            "predictions": predictions.tolist(),
            "total_rows": len(new_data),
            "predicted_rows": len(valid_indices),
            "skipped_rows": len(new_data) - len(valid_indices),
            "valid_indices": valid_indices,
            "prediction_stats": {
                "min": float(predictions.min()),
                "max": float(predictions.max()),
                "mean": float(predictions.mean()),
                "median": float(np.median(predictions))
            },
            "feature_info": {
                "required_features": len(train_features),
                "provided_features": len(present_features),
                "missing_features": len(missing_features),
                "missing_feature_names": missing_features,
                "auto_filled_with_zero": len(missing_features) > 0
            }
        })
    
    except Exception as e:
        logger.error(f"\n❌ ERROR in predict-json:")
        logger.error(f"Error: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        logger.error("=" * 60 + "\n")
        
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
    logger.error(f"500 Error: {str(e)}")
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        "error": "File too large",
        "details": "Maximum file size is 16MB"
    }), 413

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
