# Day 2: added model persistence with joblib and database integration
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import shap
import os
import json
from datetime import datetime
from typing import Optional, List, Dict
import sys

# Ensure config is importable from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    import config
except ImportError:
    # Fallback if path manipulation fails in some environments
    sys.path.append(os.getcwd())
    import config

class FraudPipeline:
    def __init__(self, data_path: str) -> None:
        """
        Initialize the Fraud Detection Pipeline.
        
        Args:
            data_path: Path to the raw CSV dataset.
        """
        self.data_path = data_path
        self.user_features = None
        self.model = None
        self.explainer = None
        self.shap_values = None
        self.logs = []
        self.raw_df = None
        self.trained_at = None
        self.model_loaded_from_disk = False
        
    def add_log(self, action: str, actor: str, detail: str) -> None:
        """
        Add a system log entry.
        
        Args:
            action: The action performed.
            actor: The entity performing the action.
            detail: Specific details about the action.
        """
        self.logs.append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Action": action,
            "Actor": actor,
            "Detail": detail
        })

    def load_model(self) -> bool:
        import pickle
        try:
            if os.path.exists(config.MODEL_PATH) and os.path.exists(config.USER_FEATURES_PATH):
                with open(config.MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
                with open(config.USER_FEATURES_PATH, 'rb') as f:
                    self.user_features = pickle.load(f)
                self.explainer = shap.TreeExplainer(self.model)
                X = self.user_features[['Return Frequency', 'High-Value Item Ratio',
                                        'Avg Time-to-Return', 'Reason Diversity',
                                        'Days Active', 'Top Reason Ratio']]
                self.shap_values = self.explainer.shap_values(X)
                self.add_log("Model Loaded", "System-ML", "Loaded saved model from disk")
                return True
            return False
        except Exception as e:
            self.add_log("Model Load Failed", "System-ML", str(e))
            return False

    def run(self) -> None:
        """
        Execute the complete ML pipeline: feature engineering, training, and explainability.
        """
        self.add_log("ML Pipeline Run", "System-ML", "Started data loading and feature engineering")
        
        # Load Raw Data
        df = pd.read_csv(self.data_path)
        df['purchase_date'] = pd.to_datetime(df['purchase_date'], errors='coerce')
        df['return_date'] = pd.to_datetime(df['return_date'], errors='coerce')
        self.raw_df = df
        
        # 1. Feature Engineering
        # Calculate behavioral stats per user
        
        # Fix Return Frequency: Only count orders older than MIN_RETURN_AGE_DAYS (30 days)
        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=config.MIN_RETURN_AGE_DAYS)
        eligible_orders = df[df['purchase_date'] <= cutoff_date]
        total_eligible_orders_per_user = eligible_orders.groupby('user_id').size().reset_index(name='eligible_orders')
        
        # Keep only actual returns
        returns_df = df[df['return_reason'].notna() & (df['return_reason'] != 'Not Returned')].copy()
        returns_df['time_to_return'] = (returns_df['return_date'] - returns_df['purchase_date']).dt.days
        
        user_grouped = returns_df.groupby('user_id')
        user_stats = []
        user_purchase_range = df.groupby('user_id')['purchase_date'].agg(['min', 'max'])
        
        for user_id, group in user_grouped:
            total_returns = len(group)
            total_refund = group['refund_amount'].sum()
            avg_time_to_return = group['time_to_return'].mean()
            
            # High value ratio
            high_value_returns = len(group[group['refund_amount'] > config.HIGH_VALUE_ITEM_THRESHOLD])
            high_value_ratio = high_value_returns / total_returns if total_returns > 0 else 0
            
            # TASK 2: Real Data Engineering Calculations
            # Days Active: Time between first and last purchase
            if user_id in user_purchase_range.index:
                days_active = (user_purchase_range.loc[user_id, 'max'] - 
                               user_purchase_range.loc[user_id, 'min']).days
            else:
                days_active = 0
                
            # Reason Diversity: Unique reasons / Total returns
            return_reason_diversity = len(group['return_reason'].unique()) / total_returns
            
            # Top Reason Ratio: Count of most frequent reason / Total returns
            top_reason_ratio = group['return_reason'].value_counts().iloc[0] / total_returns
            
            user_stats.append({
                'user_id': user_id,
                'Total Returns': total_returns,
                'Financial Exposure ($)': float(total_refund),
                'Avg Time-to-Return': float(avg_time_to_return),
                'High-Value Item Ratio': float(high_value_ratio * 100),
                'Days Active': days_active,
                'Reason Diversity': float(return_reason_diversity),
                'Top Reason Ratio': float(top_reason_ratio)
            })
            
        user_df = pd.DataFrame(user_stats)
        
        # Merge with eligible orders for Return Frequency
        user_df = pd.merge(user_df, total_eligible_orders_per_user, on='user_id', how='left')
        
        # Handle cases where user has zero returns but might have eligible orders
        # (Though current loop is over returns_df, we should include all users from df)
        user_df = user_df.copy()
            
        user_df['Total Returns'] = user_df['Total Returns'].fillna(0)
        user_df['Financial Exposure ($)'] = user_df['Financial Exposure ($)'].fillna(0)
        user_df['eligible_orders'] = user_df['eligible_orders'].fillna(0)
        
        # Calculate Return Frequency based on eligible orders (older than 30 days)
        # Add comment explaining business logic
        # Business Logic: We only count orders older than 30 days in the denominator 
        # because recent orders haven't had enough time to be returned, which would artificially lower the frequency.
        user_df['Return Frequency'] = (user_df['Total Returns'] / user_df['eligible_orders']) * 100
        user_df['Return Frequency'] = user_df['Return Frequency'].replace([np.inf, -np.inf], 0).fillna(0)
        
        feature_cols = ['Return Frequency', 'High-Value Item Ratio', 'Avg Time-to-Return', 
                        'Reason Diversity', 'Days Active', 'Top Reason Ratio']
        
        # Fill NAs for all features
        user_df[feature_cols] = user_df[feature_cols].fillna(0)
        
        # 2. Anomaly Detection (Isolation Forest)
        X = user_df[feature_cols]
        self.model = IsolationForest(contamination=config.CONTAMINATION_RATE, random_state=42, n_estimators=100)
        user_df['Anomaly Score Raw'] = self.model.fit_predict(X)
        
        # Decision function to calculate score. Lower means more anomalous.
        raw_scores = self.model.decision_function(X)
        
        # Normalize to 0-100 where 100 is highly anomalous (fraudulent)
        min_score, max_score = raw_scores.min(), raw_scores.max()
        normalized_scores = config.MAX_RISK_SCORE - ((raw_scores - min_score) / (max_score - min_score) * config.MAX_RISK_SCORE)
        user_df['Risk Score'] = np.round(normalized_scores, 2)
        
        # Generate initial risk bands
        user_df['Risk Band'] = 'Low'
        user_df.loc[user_df['Risk Score'] >= config.MEDIUM_RISK_THRESHOLD, 'Risk Band'] = 'Medium'
        user_df.loc[user_df['Risk Score'] >= config.HIGH_RISK_THRESHOLD, 'Risk Band'] = 'High'
        
        # 3. Explainability (SHAP)
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer.shap_values(X)
        
        self.user_features = user_df
        
        import pickle
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        with open(config.MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)
        with open(config.USER_FEATURES_PATH, 'wb') as f:
            pickle.dump(self.user_features, f)

        # Persist to database
        try:
            from backend import database
            database.save_risk_profiles(user_df)
            high_count = int((user_df['Risk Band'] == 'High').sum())
            med_count = int((user_df['Risk Band'] == 'Medium').sum())
            low_count = int((user_df['Risk Band'] == 'Low').sum())
            database.save_pipeline_run(
                total_users=len(user_df),
                high_risk=high_count,
                medium_risk=med_count,
                low_risk=low_count,
                avg_score=float(user_df['Risk Score'].mean()),
                total_exposure=float(user_df['Financial Exposure ($)'].sum())
            )
            database.add_audit_log("Pipeline Complete", "System-ML", f"Scored {len(user_df)} users. High risk: {high_count}")
        except Exception as e:
            print(f"DB persistence error: {e}")

        self.add_log("ML Pipeline Run", "System-ML", f"Processed {len(df)} transactions. Scored {len(user_df)} users.")
        
    def get_user_profile(self, user_id_or_order_id: str) -> Optional[dict]:
        """
        Retrieve complete risk profile for a user by user_id or order_id.
        
        Args:
            user_id_or_order_id: Either a user ID (USER00000001) or order ID (ORD00000001)
        
        Returns:
            Dictionary with risk score, SHAP breakdown, and user details.
            Returns None if user not found or model not initialized.
        """
        # Input Validation
        if not user_id_or_order_id or not isinstance(user_id_or_order_id, str):
            return {"error": "Invalid input"}
        user_id_or_order_id = user_id_or_order_id.strip()

        if self.user_features is None:
            return None
            
        if self.user_features is not None:
            user_row = self.user_features[self.user_features['user_id'] == user_id_or_order_id]
            if user_row.empty:
                # Check if they exist in raw data at all
                if self.raw_df is not None:
                    raw_row = self.raw_df[self.raw_df['user_id'] == user_id_or_order_id]
                    if not raw_row.empty:
                        return {
                            "User ID": user_id_or_order_id,
                            "Risk Score": 0.0,
                            "Risk Band": "Low",
                            "Total Returns": 0,
                            "Financial Exposure ($)": 0.0,
                            "Days Active": 0,
                            "Reason Diversity": 0.0,
                            "SHAP": {"Feature": [], "Contribution": [], 
                                     "Direction": [], "Abs_Contribution": []}
                        }

        # Try matching by user_id first
        user_row = self.user_features[self.user_features['user_id'] == user_id_or_order_id]
        
        # If not found, try matching by Order ID in the raw data
        if user_row.empty and self.raw_df is not None:
            matching_order = self.raw_df[self.raw_df['order_id'] == user_id_or_order_id]
            if not matching_order.empty:
                actual_user_id = matching_order.iloc[0]['user_id']
                user_row = self.user_features[self.user_features['user_id'] == actual_user_id]

        if user_row.empty:
            return None
            
        idx = user_row.index[0]
        actual_user_id = user_row.iloc[0]['user_id']
        
        feature_cols = ['Return Frequency', 'High-Value Item Ratio', 'Avg Time-to-Return', 
                        'Reason Diversity', 'Days Active', 'Top Reason Ratio']
        
        # TASK 4: Fix SHAP sign logic
        # For Isolation Forest, negative SHAP = pushes toward anomaly = increases risk
        if self.shap_values is not None and idx < len(self.shap_values):
            shap_contributions = self.shap_values[idx] 
        else:
            shap_contributions = np.zeros(len(feature_cols))
        
        shap_data = {
            "Feature": feature_cols,
            "Contribution": np.round(shap_contributions, 2).tolist(),
            "Direction": ['increases_risk' if v < 0 else 'decreases_risk' for v in shap_contributions.tolist()],
            "Abs_Contribution": np.abs(np.round(shap_contributions, 2)).tolist()
        }
        
        return {
            "User ID": actual_user_id,
            "Risk Score": float(user_row.iloc[0]['Risk Score']),
            "Risk Band": user_row.iloc[0]['Risk Band'],
            "Total Returns": int(user_row.iloc[0]['Total Returns']),
            "Financial Exposure ($)": float(user_row.iloc[0]['Financial Exposure ($)']),
            "Days Active": int(user_row.iloc[0]['Days Active']),
            "Reason Diversity": float(user_row.iloc[0]['Reason Diversity']),
            "SHAP": shap_data
        }

    def get_user_timeline(self, user_id_or_order_id: str) -> list:
        """
        Retrieve transaction history timeline for a user.
        
        Args:
            user_id_or_order_id: Either a user ID (USER00000001) or order ID (ORD00000001)
            
        Returns:
            List of dictionaries representing purchase and return events.
        """
        # Input Validation
        if not user_id_or_order_id or not isinstance(user_id_or_order_id, str):
            return []
        user_id_or_order_id = user_id_or_order_id.strip()

        if self.raw_df is None:
            return []
            
        actual_user_id = user_id_or_order_id
        # Check if it's an order ID
        if not user_id_or_order_id.startswith('USER'):
            matching_order = self.raw_df[self.raw_df['order_id'] == user_id_or_order_id]
            if not matching_order.empty:
                actual_user_id = matching_order.iloc[0]['user_id']

        user_events = self.raw_df[self.raw_df['user_id'] == actual_user_id]
        
        timeline = []
        for _, row in user_events.iterrows():
            timeline.append({
                "Date": row['purchase_date'].strftime('%Y-%m-%d') if pd.notna(row['purchase_date']) else "Unknown",
                "Event Type": "Purchase",
                "Amount": f"${row['refund_amount']:,.2f}" if pd.notna(row['refund_amount']) else "$0.00",
                "Item Category": "General", 
                "Status": "Completed",
                "Flag": "None"
            })
            
            if pd.notna(row['return_reason']) and row['return_reason'] != 'Not Returned':
                timeline.append({
                    "Date": row['return_date'].strftime('%Y-%m-%d') if pd.notna(row['return_date']) else "Unknown",
                    "Event Type": "Return Request",
                    "Amount": f"${row['refund_amount']:,.2f}",
                    "Item Category": "General",
                    "Status": "Refunded",
                    "Flag": f"⚠️ {row['return_reason']}"
                })
        
        try:
            timeline = sorted(timeline, key=lambda x: pd.to_datetime(x['Date'], errors='coerce') if pd.notna(x['Date']) else pd.Timestamp.min, reverse=True)
        except Exception:
             pass 
             
        return timeline
        
    def get_dashboard_data(self) -> dict:
        """
        Retrieve high-level dashboard metrics and logs.
        
        Returns:
            Dictionary containing user statistics and system logs.
        """
        if self.user_features is None:
            return {}
            
        return {
            "users": self.user_features[['user_id', 'Risk Score', 'Risk Band', 'Financial Exposure ($)', 'Total Returns']].to_dict('records'),
            "logs": self.logs
        }
    
    def load_from_disk(self) -> bool:
        """
        Load a previously trained model from disk instead of retraining.
        Returns True if successful, False if no saved model exists or load fails.
        This reduces startup time from ~30 seconds to ~1 second.
        """
        try:
            if not os.path.exists(config.MODEL_PATH):
                return False
            self.model = joblib.load(config.MODEL_PATH)
            self.explainer = joblib.load(config.MODEL_PATH.replace('.pkl', '_explainer.pkl'))
            self.shap_values = joblib.load(config.MODEL_PATH.replace('.pkl', '_shap.pkl'))
            self.user_features = pd.read_pickle(config.USER_FEATURES_PATH)
            self.raw_df = pd.read_pickle(config.MODEL_PATH.replace('isolation_forest.pkl', 'raw_df.pkl'))
            mtime = os.path.getmtime(config.MODEL_PATH)
            self.trained_at = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            self.model_loaded_from_disk = True
            self.add_log("Model Loaded", "System", f"Loaded trained model from disk. Trained at: {self.trained_at}")
            return True
        except Exception as e:
            print(f"Model load error: {e}")
            return False
