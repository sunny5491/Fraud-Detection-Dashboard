import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="RevGuard: Explainable Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MOCK DATA GENERATOR ---
@st.cache_data
def generate_mock_data():
    """Generates mock user fraud risk data for the dashboard."""
    np.random.seed(42)
    users = [f"USR-{np.random.randint(10000, 99999)}" for _ in range(500)]
    
    # Generate beta-distributed scores to simulate realistic (mostly low risk) distribution
    scores = np.random.beta(a=1.5, b=5, size=500) * 100
    
    data = []
    for i, user in enumerate(users):
        score = scores[i]
        
        # Determine bands based on score
        if score > 70:
            band = "High"
            action = "Immediate Investigation"
        elif score > 40:
            band = "Medium"
            action = "Manual Review"
        else:
            band = "Low"
            action = "Standard Monitoring"
            
        # Correlate returns and financial exposure with risk score
        returns = int((score / 100) * np.random.randint(10, 50)) + np.random.randint(1, 5)
        value = returns * np.random.uniform(50, 500)
        
        data.append({
            "User ID": user,
            "Risk Score": round(score, 1),
            "Risk Band": band,
            "Action Strategy": action,
            "Total Returns": returns,
            "Financial Exposure ($)": round(value, 2)
        })
    return pd.DataFrame(data)

# Load data
df = generate_mock_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🛡️ RevGuard")
st.sidebar.caption("Returns Fraud Detection Engine")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation", 
    ["📊 Risk Overview", "🔍 User Investigation", "📈 Behavioral Heatmap", "📁 Data Ingestion", "📋 Audit Logs"]
)

st.sidebar.divider()
st.sidebar.markdown("### ⚙️ Engine Settings")
risk_threshold = st.sidebar.slider(
    "High-Risk Threshold", 
    min_value=50, max_value=90, value=71, step=1,
    help="Users with scores above this threshold will be flagged as High Risk."
)
medium_threshold = st.sidebar.slider(
    "Medium-Risk Threshold",
    min_value=20, max_value=risk_threshold-1, value=41, step=1
)

st.sidebar.divider()
st.sidebar.caption("Current DB Status: **Connected** 🟢")
st.sidebar.caption("Last ML Pipeline Run: **Today, 10:45 AM**")

# Dynamically update bands based on thresholds set in sidebar
df.loc[df["Risk Score"] >= risk_threshold, "Risk Band"] = "High"
df.loc[(df["Risk Score"] >= medium_threshold) & (df["Risk Score"] < risk_threshold), "Risk Band"] = "Medium"
df.loc[df["Risk Score"] < medium_threshold, "Risk Band"] = "Low"


# ==========================================
# PAGE 1: RISK OVERVIEW
# ==========================================
if page == "📊 Risk Overview":
    st.title("📊 Enterprise Risk Overview")
    st.markdown("Monitor real-time e-commerce return fraud metrics, exposure distributions, and system health.")
    
    # KPIS
    col1, col2, col3, col4 = st.columns(4)
    total_tx = 15420
    high_risk_df = df[df["Risk Band"] == "High"]
    high_risk_users = len(high_risk_df)
    total_loss = high_risk_df["Financial Exposure ($)"].sum()
    loss_recovered = total_loss * 0.45 # mock 45% recovery representation
    
    col1.metric("Total Active Users", f"{len(df):,}")
    col2.metric("High Risk Users (Flagged)", f"{high_risk_users}", f"Threshold: {risk_threshold}")
    col3.metric("Fraud Financial Exposure", f"${total_loss:,.2f}", "-12% vs last month", delta_color="inverse")
    col4.metric("Loss Recovered (Blocked)", f"${loss_recovered:,.2f}", "+5% vs last month")
    
    st.divider()
    
    col_chart1, col_chart2 = st.columns([1, 1])
    
    with col_chart1:
        st.subheader("Risk Band Distribution")
        # Pie chart for risk bands
        fig_pie = px.pie(
            df, names='Risk Band', 
            color='Risk Band',
            color_discrete_map={'High':'#FF4B4B', 'Medium':'#FFA421', 'Low':'#00CC96'},
            hole=0.4
        )
        fig_pie.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_chart2:
        st.subheader("Top High-Risk Users")
        top_risk_users = high_risk_df.sort_values(by="Risk Score", ascending=False).head(8)
        
        # Format dataframe for display
        display_df = top_risk_users[["User ID", "Risk Score", "Financial Exposure ($)", "Total Returns"]].copy()
        display_df["Financial Exposure ($)"] = display_df["Financial Exposure ($)"].apply(lambda x: f"${x:,.2f}")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)


# ==========================================
# PAGE 2: USER INVESTIGATION
# ==========================================
elif page == "🔍 User Investigation":
    st.title("🔍 User Risk Investigation")
    st.markdown("Search a specific user's behavioral patterns, SHAP explanation factors, and audit historical actions.")
    
    search_query = st.text_input("Search User ID or Email (e.g., USR-45129, USR-82341)", "")
    
    st.divider()
    
    if search_query:
        # Check if user exists in our mock DB
        user_match = df[df["User ID"].str.contains(search_query, case=False)]
        
        if not user_match.empty:
            user_data = user_match.iloc[0]
            st.subheader(f"Profile: {user_data['User ID']}")
            
            score = user_data["Risk Score"]
            band = user_data["Risk Band"]
            
            if band == "High":
                color_hex = "#FF4B4B"
            elif band == "Medium":
                color_hex = "#FFA421"
            else:
                color_hex = "#00CC96"
            
            col_profile, col_shap = st.columns([1, 2])
            
            with col_profile:
                st.markdown(f"### AI Risk Score: <span style='color:{color_hex};'>{score}</span> / 100", unsafe_allow_html=True)
                st.markdown(f"**Risk Band:** {band}")
                st.markdown(f"**Financial Exposure:** ${user_data['Financial Exposure ($)']:,.2f}")
                st.markdown(f"**Total Returns:** {user_data['Total Returns']}")
                st.markdown("**Account Age:** 142 days")
                st.markdown("**Region:** North America (IP Mismatch Detected)")
                
                st.write("")
                st.button("🚫 Block Refund & Flag Account", type="primary", use_container_width=True)
                st.button("✅ Mark as Safe (False Positive)", use_container_width=True)
                
            with col_shap:
                st.subheader("Why was this user flagged? (SHAP Explainability)")
                
                # Mock SHAP data based on score severity
                if band == "High":
                    shap_data = {
                        "Feature": ["Return Frequency", "High-Value Item Ratio", "Avg Time-to-Return", "Geolocation Mismatch", "Account Age", "Category Repetition"],
                        "Contribution": [32.4, 18.2, 12.1, 14.5, -8.3, 5.1]
                    }
                elif band == "Medium":
                    shap_data = {
                        "Feature": ["Return Frequency", "High-Value Item Ratio", "Avg Time-to-Return", "Geolocation Mismatch", "Account Age", "Category Repetition"],
                        "Contribution": [15.1, 12.5, 4.2, 0.0, -12.4, 2.1]
                    }
                else:
                    shap_data = {
                        "Feature": ["Return Frequency", "High-Value Item Ratio", "Avg Time-to-Return", "Geolocation Mismatch", "Account Age", "Category Repetition"],
                        "Contribution": [4.1, 2.5, -1.2, 0.0, -22.4, -3.1]
                    }
                
                shap_df = pd.DataFrame(shap_data)
                shap_df = shap_df.sort_values(by="Contribution", ascending=True)
                
                # SHAP Bar Chart
                fig_shap = px.bar(
                    shap_df, x="Contribution", y="Feature", orientation='h',
                    color="Contribution", color_continuous_scale=["#00CC96", "#FF4B4B"],
                    title="Feature Impact Output (Isolation Forest)"
                )
                fig_shap.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig_shap, use_container_width=True)
                
            st.divider()
            st.subheader("Behavioral Timeline")
            
            # Mock historical data
            timeline_data = pd.DataFrame({
                "Date": [(datetime.now() - timedelta(days=x)).strftime("%Y-%m-%d") for x in [1, 5, 8, 12, 14]],
                "Event Type": ["Return Request", "Purchase", "Return Request", "Return Request", "Purchase"],
                "Amount": ["$540.00", "$540.00", "$120.50", "$899.99", "$1,560.49"],
                "Item Category": ["Electronics", "Electronics", "Apparel", "Luxury Watches", "Mixed Cart"],
                "Status": ["Pending Risk Review", "Completed", "Refunded", "Refunded", "Completed"],
                "Flag": ["⚠️ Geolocation Mismatch", "None", "None", "None", "None"]
            })
            st.table(timeline_data)
            
        else:
            st.warning("User not found in system. Please check the ID or email.")


# ==========================================
# PAGE 3: BEHAVIORAL HEATMAP
# ==========================================
elif page == "📈 Behavioral Heatmap":
    st.title("📈 Advanced Analytics & Heatmap")
    st.markdown("Cluster-level visualization of multivariate fraud patterns across user segments.")
    
    st.subheader("Risk Distribution: Return Velocity vs. High-Value Ratio")
    
    # Mock scatter data for heatmap representation
    np.random.seed(15)
    
    # Generate background scatter points
    scatter_df = pd.DataFrame({
        "Velocity (Returns in 30d)": np.random.normal(loc=4, scale=3, size=300).clip(0, 15),
        "High-Value Ratio (%)": np.random.uniform(0, 100, size=300),
    })
    
    # Create artificial fraud clusters (high velocity + high value ratio)
    scatter_df["Base Risk"] = (scatter_df["Velocity (Returns in 30d)"] * 4) + (scatter_df["High-Value Ratio (%)"] * 0.4)
    # Add noise
    scatter_df["Final Score"] = scatter_df["Base Risk"] + np.random.normal(0, 10, 300)
    scatter_df["Final Score"] = scatter_df["Final Score"].clip(0, 100)
    
    # Plotly Scatter
    fig_scatter = px.scatter(
        scatter_df, 
        x="Velocity (Returns in 30d)", 
        y="High-Value Ratio (%)", 
        color="Final Score", 
        size="Final Score",
        hover_data=["Final Score"],
        color_continuous_scale="RdYlGn_r", # Reverse Red, Yellow, Green
        opacity=0.8
    )
    
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.info("💡 **Insight:** Notice the dense clustering of deep red points in the top right quadrant. The Isolation Forest model inherently isolates these out-of-bounds behavioral clusters rapidly.")


# ==========================================
# PAGE 4: DATA INGESTION
# ==========================================
elif page == "📁 Data Ingestion":
    st.title("📁 Data Ingestion Pipeline")
    st.markdown("Upload raw transaction and return logs (.csv format) to execute the ML pipeline, compute features, and generate anomaly scores.")
    
    st.info("Expected Schema: `user_id`, `item_id`, `item_category`, `item_value`, `purchase_date`, `return_date`, `return_reason`, `ip_address`")
    
    uploaded_file = st.file_uploader("Drop your CSV files here", type=["csv"])
    
    if uploaded_file is not None:
        file_details = {"Filename": uploaded_file.name, "File size": f"{uploaded_file.size / 1024:.2f} KB"}
        st.write(file_details)
        
        if st.button("Trigger ML Pipeline Start"):
            with st.spinner("1️⃣ Validating Schema & Cleaning Data..."):
                import time; time.sleep(1.5)
            with st.spinner("2️⃣ Running Behavioral Feature Engineering Engine..."):
                time.sleep(2)
            with st.spinner("3️⃣ Scoring with Isolation Forest..."):
                time.sleep(1.5)
            with st.spinner("4️⃣ Calculating SHAP Explanations..."):
                time.sleep(1)
            with st.spinner("5️⃣ Persisting to Database..."):
                time.sleep(1)
                
            st.success(f"✅ Pipeline Execution Successful! Processed 4,212 valid transactions. Flagged 84 new High-Risk behavior events.")


# ==========================================
# PAGE 5: AUDIT LOGS
# ==========================================
elif page == "📋 Audit Logs":
    st.title("📋 System Audit Logs")
    st.markdown("Immutable record of automated system events and investigator actions for compliance.")
    
    # Generate mock logs
    logs = pd.DataFrame({
        "Timestamp": [(datetime.now() - timedelta(minutes=i*24)).strftime("%Y-%m-%d %H:%M:%S") for i in range(15)],
        "Action": [
            "Threshold Adjusted", "User Investigation", "Refund Blocked", "Data Uploaded", "Refund Blocked",
            "User Investigation", "ML Pipeline Run", "Refund Blocked", "User Investigation", "Account Cleared",
            "Threshold Adjusted", "Refund Blocked", "User Investigation", "ML Pipeline Run", "System Reboot"
        ],
        "Actor": [
            "Admin-JL", "Inv-04 (Sarah)", "Inv-04 (Sarah)", "System-Auto", "Inv-02 (Mike)",
            "Inv-01 (Dave)", "System-ML", "Inv-04 (Sarah)", "Inv-02 (Mike)", "Inv-01 (Dave)",
            "Admin-JL", "Inv-02 (Mike)", "Inv-01 (Dave)", "System-ML", "SysAdmin"
        ],
        "Detail": [
            "Changed Risk Threshold High to 71", "Searched USR-99214", "Blocked refund for USR-99214 ($450.00)",
            "Ingested transactions_august_wk2.csv", "Blocked refund for USR-10221 ($1,200.50)",
            "Searched USR-33120", "Model inferred scores for 4,212 records", "Blocked refund for USR-88210 ($99.99)",
            "Searched USR-10221", "Marked USR-33120 as Safe. Adjusted score to 20.", "Changed Risk Threshold High to 75",
            "Blocked refund for USR-90111 ($3,400.00)", "Searched USR-33120", "Model trained on batch 8820", "Security Patch Update"
        ]
    })
    
    st.dataframe(logs, use_container_width=True, hide_index=True)
    
    col1, col2 = st.columns([8, 1])
    with col2:
        st.button("Export CSV")