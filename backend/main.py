# Day 2: added load_from_disk startup, action endpoint, pipeline-runs endpoint
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from typing import Optional, List
import uvicorn
import os
import sys

# Add parent directory to path so it can find backend and config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from backend.ml.pipeline import FraudPipeline
from backend import database

# Initialize pipeline
pipeline = FraudPipeline(config.DATA_PATH)

app = FastAPI(title="RevGuard API", description="Backend for Fraud Detection Engine")

@app.on_event("startup")
async def startup_event():
    if os.path.exists(config.DATA_PATH):
        loaded = pipeline.load_model()
        if not loaded:
            pipeline.run()

@app.get("/")
def read_root():
    return {"status": "Online", "message": "RevGuard API is running", "model_ready": pipeline.user_features is not None}

@app.get("/api/v1/risk-stats")
def get_risk_stats():
    if pipeline.user_features is None:
        raise HTTPException(status_code=503, detail="Model not initialized. Run pipeline first.")
        
    df = pipeline.user_features
    high_risk_df = df[df["Risk Band"] == "High"]
    
    total_loss = high_risk_df["Financial Exposure ($)"].sum()
    blocked_logs = [log for log in pipeline.logs 
                    if log.get("Action") == "Refund Blocked"]
    blocked_user_ids = []
    for log in blocked_logs:
        detail = log.get("Detail", "")
        for word in detail.split():
            if word.startswith("USER"):
                blocked_user_ids.append(word.rstrip("."))
    if blocked_user_ids and pipeline.user_features is not None:
        blocked_df = pipeline.user_features[
            pipeline.user_features['user_id'].isin(blocked_user_ids)
        ]
        recovered_losses = float(blocked_df["Financial Exposure ($)"].sum())
    else:
        recovered_losses = 0.0
    
    return {
        "total_users": len(df),
        "high_risk_flagged": len(high_risk_df),
        "financial_exposure": float(total_loss),
        "recovered_losses": float(recovered_losses)
    }

@app.get("/api/v1/users")
def get_users(query: Optional[str] = None):
    if pipeline.user_features is None:
        raise HTTPException(status_code=503, detail="Model not initialized. Run pipeline first.")
        
    df = pipeline.user_features
    # Simple search
    if query:
        query = query.strip()
        df = df[df['user_id'].str.contains(query, case=False, na=False)]
        
    data = df[['user_id', 'Risk Score', 'Risk Band', 'Financial Exposure ($)', 'Total Returns']].head(50).to_dict('records')
    # Rename keys
    renamed = []
    for row in data:
        renamed.append({
            "User ID": row["user_id"],
            "Risk Score": float(row["Risk Score"]),
            "Risk Band": row["Risk Band"],
            "Financial Exposure ($)": float(row["Financial Exposure ($)"]),
            "Total Returns": int(row["Total Returns"])
        })
    return renamed

@app.get("/api/v1/users/all-bands")
def get_all_user_bands():
    if pipeline.user_features is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    df = pipeline.user_features
    return df[['user_id', 'Risk Band']].to_dict('records')

@app.get("/api/v1/users/{user_id}")
def get_user_details(user_id: str):
    if not user_id or not isinstance(user_id, str):
        raise HTTPException(status_code=400, detail="Invalid User ID or Order ID")
    
    user_id = user_id.strip()
    profile = pipeline.get_user_profile(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="User not found")
        
    timeline = pipeline.get_user_timeline(user_id)
    profile["timeline"] = timeline
    
    return profile

@app.get("/api/v1/heatmap-data")
def get_heatmap_data():
    if pipeline.user_features is None:
        return []
        
    df = pipeline.user_features
    # Return features used in Behavioral Analytics
    cols = ['user_id', 'Return Frequency', 'High-Value Item Ratio',
            'Avg Time-to-Return', 'Reason Diversity', 'Top Reason Ratio',
            'Days Active', 'Risk Score', 'Risk Band']
    return df[cols].head(500).to_dict('records')

@app.post("/api/v1/run-pipeline")
def run_pipeline(background_tasks: BackgroundTasks):
    background_tasks.add_task(pipeline.run)
    return {"status": "success", "message": "ML Pipeline triggered in background"}

@app.get("/api/v1/pipeline-runs")
def get_pipeline_runs():
    return database.get_pipeline_runs()

@app.post("/api/v1/users/{user_id}/action")
def record_investigator_action(user_id: str, action_type: str, analyst_name: str, notes: str = None):
    if action_type not in ['blocked', 'cleared', 'noted']:
        raise HTTPException(status_code=400, detail="action_type must be: blocked, cleared, or noted")
    if not analyst_name or not analyst_name.strip():
        raise HTTPException(status_code=400, detail="analyst_name is required")
    
    override_status = 'Blocked' if action_type == 'blocked' else 'Cleared' if action_type == 'cleared' else None
    
    database.save_investigator_action(user_id, action_type, analyst_name.strip(), notes)
    if override_status:
        database.update_user_override(user_id, override_status, analyst_name.strip(), notes)
    
    return {"status": "success", "user_id": user_id, "action": action_type, "analyst": analyst_name}

@app.get("/api/v1/logs")
def get_logs():
    return database.get_audit_logs()

if __name__ == "__main__":
    uvicorn.run("backend.main:app" if __package__ else "main:app", host="0.0.0.0", port=8001, reload=True)