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

# Configure logging with maximum verbosity
logging.basicConfig(
    level=logging.DEBUG,
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
        logger.info(f"Current directory: {os.getcwd()}")
        logger.info(f"__file__ is: {__file__}")
        logger.info(f"Directory of __file__: {os.path.dirname(__file__)}")
        
        if not os.path.exists(MODEL_PATH):
            logger.error(f"❌ Model file not found at {MODEL_PATH}")
            # Try to find it
            search_paths = [
                "lightgbm_model_optimized.pkl",
                "./lightgbm_model_optimized.pkl",
                "../lightgbm_model_optimized.pkl",
                "/opt/render/project/src/lightgbm_model_optimized.pkl",
                "/opt/render/project/src/backend/lightgbm_model_optimized.pkl"
            ]
            logger.info("Searching for model in alternate locations...")
            for path in search_paths:
                if os.path.exists(path):
                    logger.info(f"Found model at: {path}")
                    MODEL_PATH = path
                    break
            else:
                logger.error("Model file not found in any location")
                return False
        
        logger.info(f"Loading model from {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        logger.info(f"✅ Model loaded successfully")
        logger.info(f"Model type: {type(model)}")
        
        # Test getting features
        try:
            features = model.booster_.feature_name()
            logger.info(f"Model has {len(features)} features")
            logger.info(f"First 5 features: {features[:5]}")
        except Exception as e:
            logger.error(f"Could not get feature names: {str(e)}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# Load model on startup
logger.info("=" * 80)
logger.info("STARTING WESM PRICE PREDICTION API v3.1 (DEBUG MODE)")
logger.info("=" * 80)
model_loaded = load_model()
logger.info(f"Model loaded: {model_loaded}")
logger.info("=" * 80)

def prepare_data_for_prediction(new_data):
    """
    Prepare data for prediction with extensive debugging
    """
    try:
        logger.info("\n" + "="*60)
        logger.info("STARTING DATA PREPARATION")
        logger.info("="*60)
        
        if model is None:
            logger.error("❌ Model is None!")
            raise ValueError("Model is not loaded")
        
        logger.info(f"✅ Model is loaded: {type(model)}")
        logger.info(f"📊 Input data shape: {new_data.shape}")
        logger.info(f"📊 Input columns ({len(new_data.columns)}): {list(new_data.columns)}")
        logger.info(f"📊 Input dtypes:\n{new_data.dtypes}")
        logger.info(f"📊 First row sample:\n{new_data.head(1).to_dict('records')}")
        
        # Get training features
        logger.info("\n--- Getting Model Features ---")
        try:
            train_features = model.booster_.feature_name()
            logger.info(f"✅ Got {len(train_features)} training features")
            logger.info(f"Training features: {train_features}")
        except Exception as e:
            logger.error(f"❌ Failed to get training features: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        # Check for missing features
        logger.info("\n--- Checking Missing Features ---")
        missing_features = [f for f in train_features if f not in new_data.columns]
        present_features = [f for f in train_features if f in new_data.columns]
        
        logger.info(f"Present features ({len(present_features)}): {present_features}")
        logger.info(f"Missing features ({len(missing_features)}): {missing_features}")
        
        # Fill missing features
        if missing_features:
            logger.info(f"\n--- Filling {len(missing_features)} Missing Features with 0 ---")
            for feature in missing_features:
                new_data[feature] = 0
                logger.debug(f"Filled '{feature}' with 0")
            logger.info("✅ All missing features filled")
        
        # Select features in correct order
        logger.info("\n--- Selecting Features ---")
        try:
            X_new = new_data[train_features].copy()
            logger.info(f"✅ Selected features, shape: {X_new.shape}")
            logger.info(f"X_new columns: {list(X_new.columns)}")
            logger.info(f"X_new dtypes:\n{X_new.dtypes.value_counts()}")
        except Exception as e:
            logger.error(f"❌ Failed to select features: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        # Handle categorical columns
        logger.info("\n--- Converting Categorical Columns ---")
        object_cols = X_new.select_dtypes(include=["object"]).columns.tolist()
        logger.info(f"Object columns to convert ({len(object_cols)}): {object_cols}")
        
        for col in object_cols:
            logger.debug(f"Converting '{col}' to category")
            X_new[col] = X_new[col].astype("category")
        
        logger.info("✅ Categorical conversion complete")
        logger.info(f"Final dtypes:\n{X_new.dtypes.value_counts()}")
        
        # Check for NaN values
        logger.info("\n--- Checking for NaN Values ---")
        nan_counts = X_new.isnull().sum()
        cols_with_nan = nan_counts[nan_counts > 0]
        
        if len(cols_with_nan) > 0:
            logger.warning(f"⚠️ Found NaN values in {len(cols_with_nan)} columns:")
            for col, count in cols_with_nan.items():
                logger.warning(f"  {col}: {count} NaN values")
        else:
            logger.info("✅ No NaN values found")
        
        # Get clean rows
        logger.info("\n--- Filtering Valid Rows ---")
        valid_mask = X_new.notnull().all(axis=1)
        X_new_clean = X_new[valid_mask]
        valid_indices = X_new_clean.index.tolist()
        
        logger.info(f"Valid rows: {len(valid_indices)} out of {len(new_data)}")
        logger.info(f"Skipped rows: {len(new_data) - len(valid_indices)}")
        
        if len(valid_indices) == 0:
            logger.error("❌ No valid rows! All rows have NaN values")
            logger.error("NaN summary by column:")
            logger.error(X_new.isnull().sum().to_string())
            return new_data, np.array([]), []
        
        logger.info(f"✅ Clean data shape: {X_new_clean.shape}")
        
        # Make predictions
        logger.info("\n--- Making Predictions ---")
        logger.info(f"Input to model: shape={X_new_clean.shape}, dtypes={X_new_clean.dtypes.value_counts().to_dict()}")
        
        try:
            logger.info("Calling model.predict()...")
            predicted_prices = model.predict(X_new_clean)
            logger.info(f"✅ Prediction successful!")
            logger.info(f"Predictions shape: {predicted_prices.shape}")
            logger.info(f"Predictions type: {type(predicted_prices)}")
            logger.info(f"Predictions dtype: {predicted_prices.dtype}")
            logger.info(f"📈 Stats:")
            logger.info(f"   Min: {predicted_prices.min():.2f}")
            logger.info(f"   Max: {predicted_prices.max():.2f}")
            logger.info(f"   Mean: {predicted_prices.mean():.2f}")
            logger.info(f"   Median: {np.median(predicted_prices):.2f}")
            logger.info(f"   First 5 predictions: {predicted_prices[:5]}")
        except Exception as e:
            logger.error(f"❌ Prediction failed: {str(e)}")
            logger.error(f"X_new_clean info:")
            logger.error(f"  Shape: {X_new_clean.shape}")
            logger.error(f"  Columns: {list(X_new_clean.columns)}")
            logger.error(f"  Dtypes: {X_new_clean.dtypes.to_dict()}")
            logger.error(traceback.format_exc())
            raise
        
        # Add predictions to dataframe
        logger.info("\n--- Adding Predictions to DataFrame ---")
        result_df = new_data.copy()
        result_df["Predicted_EOD_WESM_Price"] = np.nan
        result_df.loc[valid_indices, "Predicted_EOD_WESM_Price"] = predicted_prices
        
        # Add metadata
        result_df["_prediction_status"] = "skipped_missing_values"
        result_df.loc[valid_indices, "_prediction_status"] = "predicted"
        
        logger.info("✅ Predictions added to result dataframe")
        logger.info(f"Result shape: {result_df.shape}")
        logger.info(f"Predicted rows: {(result_df['_prediction_status'] == 'predicted').sum()}")
        
        logger.info("="*60)
        logger.info("DATA PREPARATION COMPLETE")
        logger.info("="*60 + "\n")
        
        return result_df, predicted_prices, valid_indices
        
    except Exception as e:
        logger.error(f"\n❌ FATAL ERROR in prepare_data_for_prediction: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.route("/", methods=["GET"])
def home():
    """Root endpoint"""
    return jsonify({
        "service": "WESM Price Prediction API",
        "version": "3.1-DEBUG",
        "status": "running",
        "model_loaded": model is not None,
        "debug_mode": True
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
            status["sample_features"] = train_features[:10]
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
        logger.info("\n" + "=" * 80)
        logger.info("NEW PREDICT-FILE REQUEST")
        logger.info("=" * 80)
        logger.info(f"Request method: {request.method}")
        logger.info(f"Content-Type: {request.content_type}")
        logger.info(f"Content-Length: {request.content_length}")
        logger.info(f"Files: {list(request.files.keys())}")
        logger.info(f"Form: {list(request.form.keys())}")
        
        # Check model
        if model is None:
            logger.error("❌ Model not loaded")
            return jsonify({"error": "Model not loaded"}), 503
        
        logger.info("✅ Model is loaded")
        
        # Check file upload
        if "file" not in request.files:
            logger.error(f"❌ No 'file' in request. Available: {list(request.files.keys())}")
            return jsonify({
                "error": "No file uploaded",
                "received_keys": list(request.files.keys())
            }), 400
        
        file = request.files["file"]
        logger.info(f"✅ File received: {file.filename}")
        
        if not file or file.filename == "":
            logger.error("❌ Empty filename")
            return jsonify({"error": "Empty filename"}), 400
        
        # Get sheet name
        sheet_name = request.form.get("sheet_name", "0")
        try:
            sheet_name = int(sheet_name) if sheet_name.isdigit() else sheet_name
        except:
            sheet_name = 0
        
        logger.info(f"📊 Sheet name: {sheet_name}")
        
        # Read Excel file
        logger.info("\n--- Reading Excel File ---")
        try:
            file_content = file.read()
            logger.info(f"File size: {len(file_content):,} bytes")
            
            new_data = pd.read_excel(io.BytesIO(file_content), sheet_name=sheet_name)
            logger.info(f"✅ Excel read: {new_data.shape}")
            logger.info(f"Columns: {list(new_data.columns)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to read Excel: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                "error": "Failed to read Excel file",
                "details": str(e)
            }), 400
        
        if new_data.empty:
            logger.error("❌ DataFrame is empty")
            return jsonify({"error": "Uploaded file is empty"}), 400
        
        # Make predictions
        logger.info("\n--- Starting Prediction Process ---")
        result_df, predictions, valid_indices = prepare_data_for_prediction(new_data)
        
        if len(predictions) == 0:
            logger.error("❌ No predictions generated")
            return jsonify({
                "error": "No valid rows to predict",
                "details": "All rows have missing values"
            }), 400
        
        logger.info(f"✅ Generated {len(predictions)} predictions")
        
        # Create Excel output
        logger.info("\n--- Creating Excel Output ---")
        try:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                result_df.to_excel(writer, index=False, sheet_name='Predictions')
            output.seek(0)
            logger.info(f"✅ Excel created, size: {len(output.getvalue())} bytes")
        except Exception as e:
            logger.error(f"❌ Failed to create Excel: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                "error": "Failed to create output file",
                "details": str(e)
            }), 500
        
        logger.info("="*80)
        logger.info(f"✅ SUCCESS - Returning Excel file with {len(predictions)} predictions")
        logger.info("="*80 + "\n")
        
        return send_file(
            output,
            as_attachment=True,
            download_name="predicted_output.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    except Exception as e:
        logger.error(f"\n{'='*80}")
        logger.error("❌ FATAL ERROR in predict-file")
        logger.error(f"{'='*80}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Full traceback:")
        logger.error(traceback.format_exc())
        logger.error(f"{'='*80}\n")
        
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
        logger.info("\n" + "=" * 80)
        logger.info("NEW PREDICT-JSON REQUEST")
        logger.info("=" * 80)
        
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
            return jsonify({
                "success": False,
                "error": "No valid rows",
                "total_rows": len(new_data)
            }), 400
        
        # Get feature info
        train_features = model.booster_.feature_name()
        present = [f for f in train_features if f in new_data.columns]
        missing = [f for f in train_features if f not in new_data.columns]
        
        logger.info(f"✅ Returning {len(predictions)} predictions")
        
        return jsonify({
            "success": True,
            "predictions": predictions.tolist(),
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
                "missing_names": missing
            }
        })
    
    except Exception as e:
        logger.error(f"Error in predict-json: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting server on port {port}")
    logger.info(f"{'='*80}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
