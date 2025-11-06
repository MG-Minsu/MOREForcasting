from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
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

# Filter out Render health checks to reduce log noise
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
            logger.error(f"Current directory: {os.getcwd()}")
            logger.error(f"Files in directory: {os.listdir(os.path.dirname(__file__) or '.')}")
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
logger.info("Starting WESM Price Prediction API")
logger.info("=" * 60)
load_model()

def get_model_features():
    """Extract feature names from the model"""
    try:
        if hasattr(model, 'feature_name_'):
            features = model.feature_name_
            logger.info(f"Got features from feature_name_: {len(features)} features")
            return features
        elif hasattr(model, 'booster_'):
            features = model.booster_.feature_name()
            logger.info(f"Got features from booster_: {len(features)} features")
            return features
        elif hasattr(model, 'feature_names_in_'):
            features = list(model.feature_names_in_)
            logger.info(f"Got features from feature_names_in_: {len(features)} features")
            return features
        else:
            logger.warning("⚠️ Could not extract feature names from model")
            logger.warning(f"Model attributes: {dir(model)}")
            return None
    except Exception as e:
        logger.error(f"Error getting model features: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def prepare_data(df):
    """
    Prepares input dataframe for prediction
    Returns: (result_df, predictions, valid_indices)
    """
    try:
        if model is None:
            raise ValueError("Model is not loaded")
        
        logger.info(f"Preparing data: {len(df)} rows, {len(df.columns)} columns")
        logger.info(f"Input columns: {list(df.columns)[:20]}")
        
        # Get expected features from model
        expected_features = get_model_features()
        
        if expected_features is None:
            # Fallback: use all columns in dataframe
            logger.warning("⚠️ Using all dataframe columns as features (fallback)")
            expected_features = list(df.columns)
        
        # Find which features are present
        present_features = [col for col in expected_features if col in df.columns]
        missing_features = [col for col in expected_features if col not in df.columns]
        
        logger.info(f"Expected features: {len(expected_features)}")
        logger.info(f"Present features: {len(present_features)}")
        logger.info(f"Missing features: {len(missing_features)}")
        
        if len(present_features) == 0:
            raise ValueError(
                f"No training features found in uploaded file.\n"
                f"Expected: {expected_features[:10]}\n"
                f"Found: {list(df.columns)[:10]}"
            )
        
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features: {missing_features[:10]}")
        
        # Create prediction dataset
        X_pred = df[present_features].copy()
        logger.info(f"Created prediction dataset: {X_pred.shape}")
        
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
        
        # Find valid rows (no missing values)
        valid_mask = X_pred.notnull().all(axis=1)
        valid_indices = X_pred[valid_mask].index.tolist()
        
        logger.info(f"Valid rows: {len(valid_indices)} out of {len(df)}")
        
        if len(valid_indices) == 0:
            raise ValueError(
                "All rows contain missing values in required features. "
                "Please check your data and ensure all required columns have values."
            )
        
        # Get valid data
        X_valid = X_pred.loc[valid_indices]
        logger.info(f"Making predictions on {len(X_valid)} rows...")
        
        # Make predictions
        try:
            predictions = model.predict(X_valid)
            logger.info(f"✅ Predictions successful: {len(predictions)} values")
            logger.info(f"Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")
        except Exception as e:
            logger.error(f"❌ Prediction failed: {str(e)}")
            logger.error(f"X_valid shape: {X_valid.shape}")
            logger.error(f"X_valid dtypes:\n{X_valid.dtypes}")
            raise ValueError(f"Model prediction failed: {str(e)}")
        
        # Create result dataframe
        result_df = df.copy()
        result_df["Predicted_EOD_WESM_Price"] = None
        result_df.loc[valid_indices, "Predicted_EOD_WESM_Price"] = predictions
        
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
        "version": "1.1",
        "status": "running",
        "model_loaded": model is not None,
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
                status["sample_features"] = features[:10] if len(features) > 10 else features
        except Exception as e:
            status["feature_error"] = str(e)
    
    return jsonify(status), 200 if model is not None else 503

@app.route("/predict-file", methods=["POST", "OPTIONS"])
def predict_file():
    """Upload Excel and get Excel back with predictions"""
    
    # Handle OPTIONS request for CORS preflight
    if request.method == "OPTIONS":
        return "", 200
    
    try:
        logger.info("\n" + "=" * 60)
        logger.info("PREDICT-FILE REQUEST")
        logger.info("=" * 60)
        logger.info(f"Content-Type: {request.content_type}")
        logger.info(f"Content-Length: {request.content_length}")
        logger.info(f"Files: {list(request.files.keys())}")
        logger.info(f"Form: {list(request.form.keys())}")
        
        # Check model
        if model is None:
            logger.error("❌ Model not loaded")
            return jsonify({
                "error": "Model not loaded on server",
                "details": "The model file is missing or failed to load. Please contact support."
            }), 503
        
        # Check file upload
        if "file" not in request.files:
            logger.error("❌ No file in request")
            return jsonify({
                "error": "No file uploaded",
                "details": "Please include a file with the key 'file' in your form data",
                "received_keys": list(request.files.keys())
            }), 400
        
        file = request.files["file"]
        
        if not file or file.filename == "":
            logger.error("❌ Empty filename")
            return jsonify({
                "error": "Empty filename",
                "details": "The uploaded file has no name"
            }), 400
        
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
            # Read file content
            file_content = file.read()
            logger.info(f"📦 File size: {len(file_content):,} bytes")
            
            # Parse Excel
            df = pd.read_excel(io.BytesIO(file_content), sheet_name=sheet_name)
            logger.info(f"✅ Excel parsed: {len(df)} rows × {len(df.columns)} columns")
            
        except Exception as e:
            logger.error(f"❌ Failed to read Excel: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                "error": "Failed to read Excel file",
                "details": str(e),
                "type": type(e).__name__,
                "hint": "Make sure the file is a valid Excel file (.xlsx or .xls)"
            }), 400
        
        if df.empty:
            logger.error("❌ Empty dataframe")
            return jsonify({
                "error": "Uploaded file is empty",
                "details": "The Excel file contains no data"
            }), 400
        
        # Make predictions
        result_df, predictions, valid_indices = prepare_data(df)
        
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
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        logger.error("=" * 60 + "\n")
        
        return jsonify({
            "error": str(e),
            "type": type(e).__name__,
            "details": "An error occurred during prediction. Check server logs for details."
        }), 500

@app.route("/predict-json", methods=["POST", "OPTIONS"])
def predict_json():
    """Upload Excel and get predictions as JSON"""
    
    # Handle OPTIONS request for CORS preflight
    if request.method == "OPTIONS":
        return "", 200
    
    try:
        logger.info("\n" + "=" * 60)
        logger.info("PREDICT-JSON REQUEST")
        logger.info("=" * 60)
        logger.info(f"Content-Type: {request.content_type}")
        logger.info(f"Files: {list(request.files.keys())}")
        
        # Check model
        if model is None:
            logger.error("❌ Model not loaded")
            return jsonify({
                "error": "Model not loaded on server",
                "details": "The model file is missing or failed to load"
            }), 503
        
        # Check file upload
        if "file" not in request.files:
            logger.error("❌ No file in request")
            return jsonify({
                "error": "No file uploaded",
                "details": "Please include a file with the key 'file'",
                "received_keys": list(request.files.keys())
            }), 400
        
        file = request.files["file"]
        
        if not file or file.filename == "":
            logger.error("❌ Empty filename")
            return jsonify({
                "error": "Empty filename"
            }), 400
        
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
        result_df, predictions, valid_indices = prepare_data(df)
        
        logger.info(f"✅ Success! Predicted {len(predictions)} rows")
        logger.info("=" * 60 + "\n")
        
        return jsonify({
            "success": True,
            "predictions": predictions.tolist(),
            "total_rows": len(df),
            "predicted_rows": len(valid_indices),
            "skipped_rows": len(df) - len(valid_indices),
            "valid_indices": valid_indices
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

