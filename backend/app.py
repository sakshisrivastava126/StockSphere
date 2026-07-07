from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes.api.v1.metrics import router as metrics_router
from backend.routes.api.v1.forecast import router as forecast_router
from backend.routes.api.v1.model_info import router as model_info_router

# Initialize FastAPI App with clean metadata
app = FastAPI(
    title="StockSphere Demand Forecasting API",
    description=(
        "Production-quality backend layer for the StockSphere ML-powered demand forecasting "
        "and inventory optimization engine. Exposes predictions, champion model metrics, "
        "and system auditing data for the React dashboard."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configurations for future React frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, configure exact allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include versioned API routers
app.include_router(metrics_router, prefix="/api/v1", tags=["Metrics & Health Status"])
app.include_router(forecast_router, prefix="/api/v1", tags=["Demand Forecasting"])
app.include_router(model_info_router, prefix="/api/v1", tags=["Model Auditing & Details"])
