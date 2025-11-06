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
logger.info("Starting WESM Price Prediction API v2.0")
logger.info("=" * 60)
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
            logger.warning("⚠️ Could not extract feature names from model")
            return None
    except Exception as e:
        logger.error(f"Error getting model features: {str(e)}")
        return None

def prepare_data(df, fill_missing=True):
    """
    Prepares input dataframe for prediction with flexible feature handling
    
    Parameters:
    -----------
    df : DataFrame
        Input data
    fill_missing : bool
        If True, fills missing features with default values (0 for numeric, 'missing' for categorical)
        If False, raises error if features are missing
    
    Returns:
    --------
    (result_df, predictions, valid_indices)
    """
    try:
        if model is None:
            raise ValueError("Model is not loaded")
        
        logger.info(f"Preparing data: {len(df)} rows, {len(df.columns)} columns")
        logger.info(f"Input columns: {list(df.columns)}")
        
        # Get expected features from model
        expected_features = get_model_features()
        
        if expected_features is None:
            raise ValueError("Could not determine model features")
        
        logger.info(f"Model expects {len(expected_features)} features")
        
        # Find which features are present and missing
        present_features = [col for col in expected_features if col in df.columns]
        missing_features = [col for col in expected_features if col not in df.columns]
        
        logger.info(f"✅ Present features: {len(present_features)}")
        logger.info(f"❌ Missing features: {len(missing_features)}")
        
        if missing_features:
            logger.info(f"Missing feature names: {missing_features[:10]}")
        
        # Handle missing features
        if len(missing_features) > 0:
            if not fill_missing:
                raise ValueError(
                    f"Missing {len(missing_features)} required features.\n"
                    f"Missing: {missing_features[:10]}\n"
                    f"Present: {present_features[:10]}"
                )
            else:
                logger.info(f"🔧 Filling {len(missing_features)} missing features with defaults")
        
        # Create prediction dataset with ALL required features
        X_pred = pd.DataFrame()
        
        # Add present features
        for col in present_features:
            X_pred[col] = df[col]
        
        # Add missing features with default values
        if fill_missing and missing_features:
            for col in missing_features:
                # Use 0 for numeric features, 'missing' for categorical
                X_pred[col] = 0
                logger.debug(f"Added missing feature '{col}' with default value 0")
        
        # Reorder columns to match model's expected order
        X_pred = X_pred[expected_features]
        
        logger.info(f"Final dataset shape: {X_pred.shape}")
        logger.info(f"Features order matches model: {list(X_pred.columns) == list(expected_features)}")
        
        # Handle categorical columns
        categorical_cols = []
        for col in X_pred.columns:
            if X_pred[col].dtype == "object":
                X_pred[col] = X_pred[col].astype(str).astype("category")
                categorical_cols.append(col)
            elif pd.api.types.is_categorical_dtype(X_pred[col]):
                X_pred[col] = X_pred[col].astype("category")
                categorical_cols.append(col)
        
        if categorical_cols:
            logger.info(f"Converted {len(categorical_cols)} categorical columns")
        
        # Find valid rows (no missing values in PRESENT features only)
        # We allow missing features to be filled, but not missing values in provided features
        valid_mask = df[present_features].notnull().all(axis=1) if present_features else pd.Series([True] * len(df))
        valid_indices = df[valid_mask].index.tolist()
        
        logger.info(f"Valid rows: {len(valid_indices)} out of {len(df)}")
        
        if len(valid_indices) == 0:
            raise ValueError(
                "All rows contain missing values in the provided features. "
                "Please ensure your data has values for the columns you're providing."
            )
        
        # Get valid data
        X_valid = X_pred.loc[valid_indices]
        
        logger.info(f"Making predictions on {len(X_valid)} rows with {X_valid.shape[1]} features...")
        logger.info(f"X_valid shape: {X_valid.shape}")
        logger.info(f"X_valid dtypes: {X_valid.dtypes.value_counts().to_dict()}")
        
        # Make predictions
        try:
            predictions = model.predict(X_valid)
            logger.info(f"✅ Predictions successful: {len(predictions)} values")
            logger.info(f"Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")
            logger.info(f"Prediction mean: {predictions.mean():.2f}")
        except Exception as e:
            logger.error(f"❌ Prediction failed: {str(e)}")
            logger.error(f"X_valid shape: {X_valid.shape}")
            logger.error(f"Expected features: {len(expected_features)}")
            logger.error(f"X_valid columns: {list(X_valid.columns)}")
            raise ValueError(f"Model prediction failed: {str(e)}")
        
        # Create result dataframe
        result_df = df.copy()
        result_df["Predicted_EOD_WESM_Price"] = None
        result_df.loc[valid_indices, "Predicted_EOD_WESM_Price"] = predictions
        
        # Add metadata columns
        result_df["_prediction_status"] = "skipped"
        result_df.loc[valid_indices, "_prediction_status"] = "predicted"
        result_df["_missing_features_filled"] = len(missing_features)
        
        logger.info("✅ Data preparation complete")
        return result_df, predictions, valid_indices
        
    except Exception as e:
        logger.error(f"❌ Error in prepare_data: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.route("/", methods=["GET"])
def home():
    """Root endpoint"""
    return jsonify({
        "service": "WESM Price Prediction API",
        "version": "2.0",
        "status": "running",
        "model_loaded": model is not None,
        "features": {
            "flexible_features": True,
            "auto_fill_missing": True,
            "description": "Missing features are automatically filled with default values"
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
            features = get_model_features()
            if features:
                status["feature_count"] = len(features)
                status["sample_features"] = features[:10] if len(features) > 10 else features
        except Exception as e:
            status["feature_error"] = str(e)
    
    return jsonify(status), 200 if model is not None else 503

@app.route("/features", methods=["GET"])
def list_features():
    """List all required features"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    features = get_model_features()
    if features is None:
        return jsonify({"error": "Could not extract features"}), 500
    
    return jsonify({
        "total_features": len(features),
        "features": features,
        "note": "Missing features will be automatically filled with default values (0 for numeric)"
    })

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
                "error": "Model not loaded on server"
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
        
        # Get fill_missing parameter (default True)
        fill_missing = request.form.get("fill_missing", "true").lower() == "true"
        logger.info(f"Fill missing features: {fill_missing}")
        
        # Read Excel file
        try:
            file_content = file.read()
            logger.info(f"📦 File size: {len(file_content):,} bytes")
            
            df = pd.read_excel(io.BytesIO(file_content), sheet_name=sheet_name)
            logger.info(f"✅ Excel parsed: {len(df)} rows × {len(df.columns)} columns")
            
        except Exception as e:
            logger.error(f"❌ Failed to read Excel: {str(e)}")
            return jsonify({
                "error": "Failed to read Excel file",
                "details": str(e)
            }), 400
        
        if df.empty:
            return jsonify({"error": "Uploaded file is empty"}), 400
        
        # Make predictions
        result_df, predictions, valid_indices = prepare_data(df, fill_missing=fill_missing)
        
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
        
        # Get fill_missing parameter (default True)
        fill_missing = request.form.get("fill_missing", "true").lower() == "true"
        
        # Read Excel file
        try:
            file_content = file.read()
            df = pd.read_excel(io.BytesIO(file_content), sheet_name=sheet_name)
            logger.info(f"✅ Excel parsed: {len(df)} rows × {len(df.columns)} columns")
            
        except Exception as e:
            logger.error(f"❌ Failed to read Excel: {str(e)}")
            return jsonify({
                "error": "Failed to read Excel file",
                "details": str(e)
            }), 400
        
        if df.empty:
            return jsonify({"error": "Uploaded file is empty"}), 400
        
        # Make predictions
        result_df, predictions, valid_indices = prepare_data(df, fill_missing=fill_missing)
        
        # Get feature info
        expected_features = get_model_features()
        present_features = [col for col in expected_features if col in df.columns]
        missing_features = [col for col in expected_features if col not in df.columns]
        
        logger.info(f"✅ Success! Predicted {len(predictions)} rows")
        logger.info("=" * 60 + "\n")
        
        return jsonify({
            "success": True,
            "predictions": predictions.tolist(),
            "total_rows": len(df),
            "predicted_rows": len(valid_indices),
            "skipped_rows": len(df) - len(valid_indices),
            "valid_indices": valid_indices,
            "feature_info": {
                "required_features": len(expected_features),
                "provided_features": len(present_features),
                "missing_features": len(missing_features),
                "missing_feature_names": missing_features,
                "auto_filled": fill_missing
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
