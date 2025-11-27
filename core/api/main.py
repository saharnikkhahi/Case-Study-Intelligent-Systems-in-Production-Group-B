from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from predict import ModelPredictor
from integrated_optimizer import IntegratedRouteOptimizer


app = FastAPI(
    title="Fleet Management & Route Optimization API",
    description="AI-Driven Route Optimization and Delay Prediction System",
    version="2.0.0"
)

predictor = None
optimizer = None


@app.on_event("startup")
async def startup_event():
    global predictor, optimizer
    try:
        models_dir = Path(__file__).parent.parent / "outputs" / "models"
        preprocessor_dir = Path(__file__).parent.parent / "outputs" / "preprocessor"
        
        predictor = ModelPredictor(
            models_dir=str(models_dir),
            preprocessor_dir=str(preprocessor_dir)
        )
        predictor.load_all_models()
        
        optimizer = IntegratedRouteOptimizer(
            models_dir=str(models_dir),
            preprocessor_dir=str(preprocessor_dir)
        )
        
        print("Models and optimizer loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load models: {e}")
        print("Run training first: python main.py")


class StopData(BaseModel):
    route_id: int
    driver_id: int
    stop_id: int
    address_id: int
    week_id: int
    country: int
    day_of_week: str
    indexp: int
    indexa: int
    arrived_time: str
    earliest_time: str
    latest_time: str
    distancep: float
    distancea: float
    depot: int
    delivery: int


class PredictionRequest(BaseModel):
    stops: List[StopData]
    model_type: Optional[str] = "random_forest"


class PredictionResponse(BaseModel):
    route_id: int
    stop_id: int
    driver_id: int
    delayed_flag_pred: int
    delay_probability: Optional[float]
    delay_minutes_pred: float


class RouteAggregate(BaseModel):
    route_id: int
    route_delay_rate: float
    avg_delay_probability: float
    total_delay_minutes: float
    avg_delay_minutes: float
    max_delay_minutes: float


@app.get("/")
async def root():
    return {
        "name": "Fleet Management API",
        "version": "1.0.0",
        "status": "running",
        "models_loaded": predictor is not None
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_available": list(predictor.models.keys()) if predictor else []
    }


@app.post("/predict/delays", response_model=List[PredictionResponse])
async def predict_delays(request: PredictionRequest):
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please run training first."
        )
    
    try:
        stops_data = [stop.dict() for stop in request.stops]
        df = pd.DataFrame(stops_data)
        
        predictions = predictor.predict_route_delays(df)
        
        results = []
        for _, row in predictions.iterrows():
            results.append(PredictionResponse(
                route_id=int(row["route_id"]),
                stop_id=int(row["stop_id"]),
                driver_id=int(row["driver_id"]),
                delayed_flag_pred=int(row["delayed_flag_pred"]),
                delay_probability=float(row["delay_probability"]) if row["delay_probability"] is not None else None,
                delay_minutes_pred=float(row["delay_minutes_pred"])
            ))
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/routes", response_model=List[RouteAggregate])
async def predict_route_aggregates(request: PredictionRequest):
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please run training first."
        )
    
    try:
        stops_data = [stop.dict() for stop in request.stops]
        df = pd.DataFrame(stops_data)
        
        route_aggregates = predictor.predict_and_aggregate_routes(df)
        
        results = []
        for _, row in route_aggregates.iterrows():
            results.append(RouteAggregate(
                route_id=int(row["route_id"]),
                route_delay_rate=float(row["route_delay_rate"]),
                avg_delay_probability=float(row["avg_delay_probability"]),
                total_delay_minutes=float(row["total_delay_minutes"]),
                avg_delay_minutes=float(row["avg_delay_minutes"]),
                max_delay_minutes=float(row["max_delay_minutes"])
            ))
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    if predictor is None:
        return {"models": [], "status": "Models not loaded"}
    
    return {
        "models": list(predictor.models.keys()),
        "optimizer_ready": optimizer is not None,
        "status": "Models loaded successfully"
    }


@app.post("/optimize/routes")
async def optimize_routes(request: PredictionRequest):
    if optimizer is None:
        raise HTTPException(
            status_code=503,
            detail="Optimizer not loaded. Please run training first."
        )
    
    try:
        stops_data = [stop.dict() for stop in request.stops]
        df = pd.DataFrame(stops_data)
        
        result = optimizer.predict_and_optimize(df, time_limit=30)
        
        if result is None:
            raise HTTPException(status_code=500, detail="Optimization failed")
        
        return {
            "solution": result['solution'],
            "num_reassignments": len(result['reassignments']),
            "reassignments": result['reassignments'][:20]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reassign/stops")
async def reassign_stops(request: PredictionRequest, delay_threshold: float = 0.7):
    if optimizer is None:
        raise HTTPException(
            status_code=503,
            detail="Optimizer not loaded. Please run training first."
        )
    
    try:
        stops_data = [stop.dict() for stop in request.stops]
        df = pd.DataFrame(stops_data)
        
        predictions = predictor.predict_route_delays(df)
        
        available_drivers = df['driver_id'].unique().tolist()
        
        routes_reassigned, reassignments = optimizer.reassignment.reassign_stops(
            df,
            predictions,
            available_drivers
        )
        
        return {
            "num_reassignments": len(reassignments),
            "reassignments": reassignments,
            "message": f"Identified {len(reassignments)} reassignment opportunities"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/realtime/adjust")
async def realtime_adjustment(
    current_routes: List[StopData],
    live_delays: List[Dict[str, Any]]
):
    if optimizer is None:
        raise HTTPException(
            status_code=503,
            detail="Optimizer not loaded."
        )
    
    try:
        current_df = pd.DataFrame([stop.dict() for stop in current_routes])
        delays_df = pd.DataFrame(live_delays)
        
        adjusted_solution = optimizer.real_time_adjustment(current_df, delays_df)
        
        return {
            "adjusted_routes": adjusted_solution,
            "message": "Routes adjusted based on live delays"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/optimization/status")
async def optimization_status():
    return {
        "predictor_loaded": predictor is not None,
        "optimizer_loaded": optimizer is not None,
        "available_models": list(predictor.models.keys()) if predictor else [],
        "optimization_ready": optimizer is not None
    }
