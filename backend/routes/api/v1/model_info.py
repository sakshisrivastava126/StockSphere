from fastapi import APIRouter, HTTPException
from backend.schemas.forecast_schema import ModelInfoResponse
from backend.services.forecasting_service import forecasting_service

router = APIRouter()

@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Get Model Information",
    description="Returns detailed champion model configurations, historical training parameters, feature count, evaluation date, and blend weight structures for auditing or dashboard visualization."
)
def get_model_info():
    try:
        return forecasting_service.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve model info: {str(e)}")
