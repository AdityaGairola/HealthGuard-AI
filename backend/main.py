# HealthGuard AI ” Backend main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.database.db import engine, Base
from backend.routes import auth_routes, profile_routes, predict_routes, dashboard_routes, model_routes, report_routes

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield

app = FastAPI(
    title="HealthGuard AI API",
    description="AI-Based Disease Risk Prediction and Health Monitoring System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_routes.router,      prefix="/api", tags=["Authentication"])
app.include_router(profile_routes.router,   prefix="/api", tags=["Health Profile"])
app.include_router(predict_routes.router,   prefix="/api", tags=["Predictions"])
app.include_router(dashboard_routes.router, prefix="/api", tags=["Dashboard"])
app.include_router(model_routes.router,     prefix="/api", tags=["Model Performance"])
app.include_router(report_routes.router,    prefix="/api", tags=["Reports"])

@app.get("/")
def root():
    return {"message": "HealthGuard AI API is running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

