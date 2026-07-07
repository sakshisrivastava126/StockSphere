from fastapi import APIRouter, HTTPException
from backend.schemas.forecast_schema import ForecastRequest, ForecastResponse
from backend.services.forecasting_service import forecasting_service

router = APIRouter()

@router.post(
    "/predict",
    response_model=ForecastResponse,
    summary="Generate Demand Forecast",
    description="Generates daily demand forecasts and inventory optimization recommendations for a custom number of days (1 to 90). The prediction horizon dynamically starts from (last_date_in_dataset + 1 day). The response returns predictions containing: predicted demand, safety stock, and recommended inventory."
)
def generate_forecast(request: ForecastRequest):
    try:
        return forecasting_service.generate_forecast(request.forecast_days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate prediction: {str(e)}")
