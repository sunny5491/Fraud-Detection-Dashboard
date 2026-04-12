from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from typing import Optional, List
import uvicorn
import os
import sys

# Add parent directory to path so it can find backend and config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from backend.ml.pipeline import FraudPipeline

# Initialize pipeline
pipeline = FraudPipeline(config.DATA_PATH)

app = FastAPI(title="RevGuard API", description="Backend for Fraud Detection Engine")

@app.on_event("startup")
async def startup_event():
    # If the initial file exists, run pipeline
    if os.path.exists(config.DATA_PATH):
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
    recovered_losses = total_loss * 0.45
    
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
    cols = ['user_id', 'Return Frequency', 'High-Value Item Ratio', 'Risk Score', 'Risk Band']
    return df[cols].head(500).to_dict('records')

@app.post("/api/v1/run-pipeline")
def run_pipeline(background_tasks: BackgroundTasks):
    background_tasks.add_task(pipeline.run)
    return {"status": "success", "message": "ML Pipeline triggered in background"}

@app.get("/api/v1/logs")
def get_logs():
    return pipeline.logs

if __name__ == "__main__":
    uvicorn.run("backend.main:app" if __package__ else "main:app", host="0.0.0.0", port=8001, reload=True)