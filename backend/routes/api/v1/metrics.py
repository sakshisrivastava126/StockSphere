from fastapi import APIRouter, HTTPException
from backend.schemas.forecast_schema import HealthResponse, MetricsResponse
from backend.services.forecasting_service import forecasting_service

router = APIRouter()

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check the system status of StockSphere's forecasting API service."
)
def get_health():
    return {"status": "healthy", "version": "1.0.0"}

@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Get Championship Model Performance Metrics",
    description="Returns pre-calculated performance metrics (WAPE, MAE, Business Cost, and Model Name) for the primary champion LightGBM forecasting model evaluated on the test dataset."
)
def get_metrics():
    try:
        return forecasting_service.get_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve model metrics: {str(e)}")
