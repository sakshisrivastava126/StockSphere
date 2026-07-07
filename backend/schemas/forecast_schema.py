from pydantic import BaseModel, Field
from typing import List, Dict, Any

class ForecastRequest(BaseModel):
    forecast_days: int = Field(
        ..., 
        gt=0, 
        le=90, 
        description="Number of days to forecast (1 to 90 days)",
        json_schema_extra={"example": 30}
    )

class PredictionItem(BaseModel):
    date: str = Field(..., description="Date of the prediction in YYYY-MM-DD format")
    predicted_demand: int = Field(..., ge=0, description="Predicted unit demand")
    safety_stock: int = Field(..., ge=0, description="Safety stock recommendation based on multiplier")
    recommended_inventory: int = Field(..., ge=0, description="Recommended total inventory (predicted demand + safety stock)")

class ForecastResponse(BaseModel):
    model: str = Field(..., description="The model or blend used to generate the forecast")
    forecast_days: int = Field(..., description="Number of days forecasted")
    predictions: List[PredictionItem] = Field(..., description="Daily forecast entries")

class MetricsResponse(BaseModel):
    wape: float = Field(..., description="Weighted Absolute Percentage Error (%)")
    mae: float = Field(..., description="Mean Absolute Error (units)")
    business_cost: float = Field(..., description="Asymmetric business inventory cost ($)")
    model_name: str = Field(..., description="Name of the primary model evaluated")

    model_config = {
        "protected_namespaces": ()
    }

class ModelInfoResponse(BaseModel):
    model_name: str = Field(..., description="Name of the model")
    wape: float = Field(..., description="Weighted Absolute Percentage Error (%)")
    feature_count: int = Field(..., description="Number of features used in the LightGBM model")
    last_trained_date: str = Field(..., description="Date the model was last trained")
    additional_metadata: Dict[str, Any] = Field(..., description="Additional configuration parameters and training statistics")

    model_config = {
        "protected_namespaces": ()
    }

class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall service status")
    version: str = Field(..., description="Service version")
