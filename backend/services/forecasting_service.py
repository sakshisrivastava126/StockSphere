import os
import sys
import json
import logging
from pathlib import Path
from datetime import timedelta
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ForecastingService")

# Locate project root and data directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Add directories to system path for importing hybrid_forecaster
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(DATA_DIR) not in sys.path:
    sys.path.append(str(DATA_DIR))

# Import the ML pipeline module
try:
    import hybrid_forecaster
except ImportError as e:
    logger.error("Failed to import hybrid_forecaster from data directory. Ensure the path is correct.")
    raise e

class ForecastingService:
    def __init__(self):
        self.gbm_model = None
        self.rf_model = None
        self.prophet_model = None
        self.meta = {}
        self.metrics = {}
        self.daily_df = None
        self.daily_featured = None
        self.last_historical_date = None
        
        # Load all components at startup
        self.load_artifacts()

    def load_artifacts(self):
        """Load trained models, metadata, dataset and metrics JSON."""
        logger.info("Initializing models and loading datasets...")
        try:
            # Load models from hybrid_forecaster module
            self.gbm_model, self.rf_model, self.prophet_model, self.meta = hybrid_forecaster.load_models()
            logger.info("ML models loaded successfully.")
            
            # Load daily aggregate transactions
            self.daily_df = hybrid_forecaster.load_unified_daily()
            # Build features for prediction
            self.daily_featured = hybrid_forecaster.build_reactive_features(self.daily_df)
            
            # Identify the last available historical date dynamically
            if not self.daily_df.empty:
                self.last_historical_date = pd.to_datetime(self.daily_df['date'].max())
                logger.info(f"Last date in historical dataset determined: {self.last_historical_date.date()}")
            else:
                raise ValueError("Historical dataset 'walmart.csv' is empty.")
                
            # Load stored metrics JSON (avoiding expensive evaluation at startup)
            metrics_path = DATA_DIR / "model_metrics.json"
            if metrics_path.exists():
                with open(metrics_path, "r") as f:
                    self.metrics = json.load(f)
                logger.info("Model metrics loaded successfully from cached file.")
            else:
                logger.warning("model_metrics.json not found in data directory. Initializing with fallback metrics.")
                self.metrics = {
                    "model_name": "LightGBM",
                    "wape": 13.55,
                    "mae": 2.0062,
                    "business_cost": 731.91,
                    "last_trained_date": self.meta.get("saved_at", "2026-04-07")[:10],
                    "feature_count": len(self.meta.get("rf_features", []))
                }
        except Exception as e:
            logger.error(f"Error loading forecasting assets: {e}")
            raise e

    def generate_forecast(self, forecast_days: int) -> dict:
        """
        Dynamically determine the start date as last_date_in_dataset + 1 day,
        generate predictions for the requested horizon, and map output with safety stock.
        """
        if self.last_historical_date is None:
            raise ValueError("Forecasting service was not initialized with historical data.")

        # Determine dynamic start and end dates
        start_date = self.last_historical_date + timedelta(days=1)
        end_date = self.last_historical_date + timedelta(days=forecast_days)
        
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        logger.info(f"Generating {forecast_days}-day forecast from {start_date_str} to {end_date_str}")
        
        try:
            # Predict using existing ML model prediction routing logic
            results = hybrid_forecaster.predict(
                start_date_str,
                end_date_str,
                self.gbm_model,
                self.prophet_model,
                self.daily_featured,
                self.meta
            )
            
            predictions = []
            for _, row in results.iterrows():
                pred_demand = int(row["final_pred"])
                rec_inv = int(row["safety_stock_order"])
                safety_stock = max(0, rec_inv - pred_demand)
                
                predictions.append({
                    "date": row["date"].strftime("%Y-%m-%d"),
                    "predicted_demand": pred_demand,
                    "safety_stock": safety_stock,
                    "recommended_inventory": rec_inv
                })
                
            model_used = results["model_used"].iloc[0] if "model_used" in results.columns else "LightGBM"
            
            return {
                "model": model_used,
                "forecast_days": forecast_days,
                "predictions": predictions
            }
        except Exception as e:
            logger.error(f"Failed to generate forecast: {e}")
            raise e

    def get_metrics(self) -> dict:
        """Returns the pre-calculated metrics for the LightGBM model."""
        return {
            "wape": self.metrics.get("wape", 13.55),
            "mae": self.metrics.get("mae", 2.0062),
            "business_cost": self.metrics.get("business_cost", 731.91),
            "model_name": self.metrics.get("model_name", "LightGBM")
        }

    def get_model_info(self) -> dict:
        """Returns model specification metadata."""
        return {
            "model_name": self.metrics.get("model_name", "LightGBM"),
            "wape": self.metrics.get("wape", 13.55),
            "feature_count": self.metrics.get("feature_count", len(self.meta.get("rf_features", []))),
            "last_trained_date": self.metrics.get("last_trained_date", self.meta.get("saved_at", "2026-04-07")[:10]),
            "additional_metadata": {
                "safety_stock_multiplier": self.meta.get("safety_stock_multiplier", 1.05),
                "blend_weights": self.meta.get("blend_weights", {"gbm": 0.7, "prophet": 0.3})
            }
        }

# Instantiate the service as a singleton
forecasting_service = ForecastingService()
