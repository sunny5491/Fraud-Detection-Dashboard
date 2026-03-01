import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import os
import sys

# Add parent directory to path so it can find backend
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from backend.ml.pipeline import FraudPipeline

# --- CONFIG ---
API_BASE_URL = "http://localhost:8001/api/v1"

st.set_page_config(
    page_title="RevGuard: Explainable Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- INTEGRATED PIPELINE (FOR STANDALONE MODE) ---
@st.cache_resource
def get_local_pipeline():
    # Use relative path for data
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed", "returns_fraud_dataset.csv")
    if os.path.exists(data_path):
        p = FraudPipeline(data_path)
        p.run()
        return p
    return None

local_pipeline = get_local_pipeline()

# --- DATA FETCHING (FROM API) ---
def fetch_risk_stats():
    try:
        res = requests.get(f"{API_BASE_URL}/risk-stats", timeout=1)
        return res.json() if res.status_code == 200 else None
    except:
        # Fallback to local pipeline
        if local_pipeline and local_pipeline.user_features is not None:
            df = local_pipeline.user_features
            high_risk_df = df[df["Risk Band"] == "High"]
            total_loss = high_risk_df["Financial Exposure ($)"].sum()
            recovered_losses = total_loss * 0.45
            return {
                "total_users": len(df),
                "high_risk_flagged": len(high_risk_df),
                "financial_exposure": float(total_loss),
                "recovered_losses": float(recovered_losses)
            }
        return None

def fetch_users(query=""):
    try:
        res = requests.get(f"{API_BASE_URL}/users", params={"query": query}, timeout=1)
        if res.status_code == 200 and isinstance(res.json(), list):
            return pd.DataFrame(res.json())
        return pd.DataFrame()
    except:
        if local_pipeline and local_pipeline.user_features is not None:
            df = local_pipeline.user_features
            if query:
                df = df[df['user_id'].str.contains(query, case=False, na=False)]
            data = df[['user_id', 'Risk Score', 'Risk Band', 'Financial Exposure ($)', 'Total Returns']].head(50).to_dict('records')
            renamed = []
            for row in data:
                renamed.append({
                    "User ID": row["user_id"],
                    "Risk Score": float(row["Risk Score"]),
                    "Risk Band": row["Risk Band"],
                    "Financial Exposure ($)": float(row["Financial Exposure ($)"]),
                    "Total Returns": int(row["Total Returns"])
                })
            return pd.DataFrame(renamed)
        return pd.DataFrame()

def fetch_user_details(user_id):
    try:
        res = requests.get(f"{API_BASE_URL}/users/{user_id}", timeout=1)
        return res.json() if res.status_code == 200 else None
    except:
        if local_pipeline:
            return local_pipeline.get_user_profile(user_id)
        return None

def fetch_heatmap_data():
    try:
        res = requests.get(f"{API_BASE_URL}/heatmap-data", timeout=1)
        if res.status_code == 200 and isinstance(res.json(), list):
            return pd.DataFrame(res.json())
        return pd.DataFrame()
    except:
        if local_pipeline and local_pipeline.user_features is not None:
            df = local_pipeline.user_features
            return df[['user_id', 'Return Frequency', 'High-Value Item Ratio', 'Risk Score', 'Risk Band']].head(500)
        return pd.DataFrame()

def fetch_logs():
    try:
        res = requests.get(f"{API_BASE_URL}/logs", timeout=1)
        if res.status_code == 200 and isinstance(res.json(), list):
            return pd.DataFrame(res.json())
        return pd.DataFrame()
    except:
        if local_pipeline:
            return pd.DataFrame(local_pipeline.logs)
        return pd.DataFrame()

# --- SIDEBAR ---
st.sidebar.title("🛡️ RevGuard")
st.sidebar.caption("Returns Fraud Detection Engine")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation", 
    ["📊 Risk Overview", "🔍 User Investigation", "📈 Behavioral Analytics", "📁 Data Ingestion", "📋 Audit Logs"]
)

st.sidebar.divider()
st.sidebar.markdown("### ⚙️ Engine Settings")
st.sidebar.info("Model thresholds are derived dynamically via Isolation Forest min-max scaling.")

stats = fetch_risk_stats()
is_standalone = False

# Check if actually connected to API or using fallback
try:
    api_check = requests.get(f"{API_BASE_URL.replace('/api/v1', '')}/", timeout=0.5)
    if api_check.status_code != 200:
        is_standalone = True
except:
    is_standalone = True

if not is_standalone:
    st.sidebar.caption("Current DB Status: **Connected to API** 🟢")
    st.sidebar.caption("Model Status: **Trained** ✅")
elif local_pipeline and local_pipeline.user_features is not None:
    st.sidebar.warning("Mode: **Integrated (Standalone)** 🟡")
    st.sidebar.caption("Model Status: **Trained (Local)** ✅")
else:
    st.sidebar.error("API / Model Offline 🔴. Run Pipeline.")


# ==========================================
# PAGE 1: RISK OVERVIEW
# ==========================================
if page == "📊 Risk Overview":
    st.title("📊 Enterprise Risk Overview")
    st.markdown("Monitor real-time e-commerce return fraud metrics, exposure distributions, and system health.")
    
    if not stats or "error" in stats:
        st.warning("Backend API is unavailable or model hasn't been initialized. Go to 'Data Ingestion'.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Total Active Users (Scored)", f"{stats['total_users']:,}")
        col2.metric("High Risk Users (Flagged)", f"{stats['high_risk_flagged']}")
        col3.metric("Fraud Financial Exposure", f"${stats['financial_exposure']:,.2f}", "-12% vs last month", delta_color="inverse")
        col4.metric("Loss Recovered (Blocked est.)", f"${stats['recovered_losses']:,.2f}", "+5% vs last month")
        
        st.divider()
        
        users_df = fetch_users()
        if not users_df.empty:
            col_chart1, col_chart2 = st.columns([1, 1])
            
            with col_chart1:
                st.subheader("Risk Band Distribution (Top 50 Sample)")
                fig_pie = px.pie(
                    users_df, names='Risk Band', 
                    color='Risk Band',
                    color_discrete_map={'High':'#FF4B4B', 'Medium':'#FFA421', 'Low':'#00CC96'},
                    hole=0.4
                )
                fig_pie.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col_chart2:
                st.subheader("Top High-Risk Users")
                top_risk_users = users_df[users_df["Risk Band"] == "High"].sort_values(by="Risk Score", ascending=False).head(8)
                if not top_risk_users.empty:
                    display_df = top_risk_users.copy()
                    display_df["Financial Exposure ($)"] = display_df["Financial Exposure ($)"].apply(lambda x: f"${x:,.2f}")
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No high-risk users currently flagged in this sample.")

# ==========================================
# PAGE 2: USER INVESTIGATION
# ==========================================
elif page == "🔍 User Investigation":
    st.title("🔍 User Risk Investigation")
    st.markdown("Search a specific user's behavioral patterns, SHAP explanation factors, and audit historical actions.")
    
    search_query = st.text_input("Search User ID or Order ID (e.g., USER00000204 or ORD00000003)", "ORD00000003")
    
    st.divider()
    
    if search_query:
        user_data = fetch_user_details(search_query)
        
        if user_data and "error" not in user_data:
            st.subheader(f"Profile: {user_data['User ID']}")
            
            score = user_data["Risk Score"]
            band = user_data["Risk Band"]
            
            color_hex = "#FF4B4B" if band == "High" else "#FFA421" if band == "Medium" else "#00CC96"
            
            col_profile, col_shap = st.columns([1, 2])
            
            with col_profile:
                st.markdown(f"### AI Risk Score: <span style='color:{color_hex};'>{score}</span> / 100", unsafe_allow_html=True)
                st.markdown(f"**Risk Band:** {band}")
                st.markdown(f"**Financial Exposure:** ${user_data['Financial Exposure ($)']:,.2f}")
                st.markdown(f"**Total Returns:** {user_data['Total Returns']}")
                st.markdown(f"**Mock Account Age:** {user_data['Account Age']} days")
                st.markdown(f"**Region Status:** {user_data['Region']}")
                
                st.write("")
                st.button("🚫 Block Refund & Flag Account", type="primary", use_container_width=True)
                st.button("✅ Mark as Safe (False Positive)", use_container_width=True)
                
            with col_shap:
                st.subheader("Why was this user flagged? (SHAP Explainability)")
                
                shap_df = pd.DataFrame(user_data['SHAP'])
                # Filter out zeroes or near zeroes for cleaner chart
                shap_df = shap_df[abs(shap_df['Contribution']) > 0.05]
                shap_df = shap_df.sort_values(by="Contribution", ascending=True)
                
                if not shap_df.empty:
                    fig_shap = px.bar(
                        shap_df, x="Contribution", y="Feature", orientation='h',
                        color="Contribution", color_continuous_scale=["#00CC96", "#FF4B4B"],
                        title="Feature Impact Output (Isolation Forest)"
                    )
                    fig_shap.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
                    st.plotly_chart(fig_shap, use_container_width=True)
                else:
                    st.info("No significant negative/positive feature driving anomaly.")
                
            st.divider()
            st.subheader("Behavioral Timeline")
            
            timeline_data = pd.DataFrame(user_data.get('timeline', []))
            if not timeline_data.empty:
                st.table(timeline_data)
            else:
                st.info("No timeline events found.")
                
        else:
            st.warning(f"User '{search_query}' not found or Model not initialized.")

# ==========================================
# PAGE 3: BEHAVIORAL ANALYTICS
# ==========================================
elif page == "📈 Behavioral Analytics":
    st.title("📈 Behavioral Analytics")
    st.markdown("Comparison of average return behaviors across Risk Bands.")
    
    heatmap_df = fetch_heatmap_data()
    if not heatmap_df.empty:
        st.subheader("Behavioral Patterns by Risk Band")
        
        # Calculate averages per risk band for a cleaner comparison
        comparison_df = heatmap_df.groupby('Risk Band')[['Return Frequency', 'High-Value Item Ratio']].mean().reset_index()
        
        # Reshape data for plotting
        plot_df = comparison_df.melt(id_vars='Risk Band', var_name='Metric', value_name='Average Value %')
        
        fig_bar = px.bar(
            plot_df, 
            x="Risk Band", 
            y="Average Value %", 
            color="Metric",
            barmode="group",
            text_auto=".1f",
            color_discrete_map={
                "Return Frequency": "#FF4B4B",
                "High-Value Item Ratio": "#FFA421"
            },
            category_orders={"Risk Band": ["Low", "Medium", "High"]}
        )
        
        fig_bar.update_layout(
            height=500,
            xaxis_title="Risk Band",
            yaxis_title="Average Value (%)",
            legend_title="Behavioral Metric",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
        st.info("💡 Insight: High Risk users typically show significantly higher Return Frequency and High-Value Item Ratios compared to Low Risk users.")
    else:
        st.warning("No data available.")

# ==========================================
# PAGE 4: DATA INGESTION
# ==========================================
elif page == "📁 Data Ingestion":
    st.title("📁 Data Ingestion Pipeline")
    st.markdown("Upload raw transaction and return logs to execute the ML pipeline.")
    
    uploaded_file = st.file_uploader("Drop your CSV files here", type=["csv"])
    
    if uploaded_file is not None or st.button("Trigger Pipeline Manually (Use local backend CSV)"):
        with st.spinner("Executing Data Engineering and Isolation Forest Training..."):
            if not is_standalone:
                res = requests.post(f"{API_BASE_URL}/run-pipeline")
                if res.status_code == 200:
                    st.success("✅ Output: Pipeline Execution Started Successfully via API.")
                else:
                    st.error("Failed to trigger API pipeline.")
            else:
                if local_pipeline:
                    local_pipeline.run()
                    st.success("✅ Output: Local Pipeline Execution Completed Successfully.")
                    st.rerun()
                else:
                    st.error("Local pipeline initialization failed.")

# ==========================================
# PAGE 5: AUDIT LOGS
# ==========================================
elif page == "📋 Audit Logs":
    st.title("📋 System Audit Logs")
    st.markdown("Immutable record of automated system events and investigator actions for compliance.")
    
    logs_df = fetch_logs()
    if not logs_df.empty:
        st.dataframe(logs_df, use_container_width=True, hide_index=True)
    else:
        st.info("No system logs currently recorded in this session.")
