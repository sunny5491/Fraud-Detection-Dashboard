# Day 2: working action buttons, sidebar improvements, force retrain, log count
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import os
import sys

# Add parent directory to path so it can find backend and config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import config
    from backend.ml.pipeline import FraudPipeline
except ImportError:
    # Handle possible import issues in some environments
    sys.path.append(os.getcwd())
    import config
    from backend.ml.pipeline import FraudPipeline

st.set_page_config(
    page_title="RevGuard: Explainable Fraud Detection",
    page_icon="Shield",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- INTEGRATED PIPELINE (FOR STANDALONE MODE) ---
@st.cache_resource
def get_local_pipeline():
    # Use relative path for data
    data_path = config.DATA_PATH
    if os.path.exists(data_path):
        p = FraudPipeline(data_path)
        p.run()
        return p
    return None

local_pipeline = get_local_pipeline()

# --- DATA FETCHING (FROM API) ---
def fetch_risk_stats():
    try:
        res = requests.get(f"{config.API_BASE_URL}/risk-stats", timeout=config.API_TIMEOUT)
        return res.json() if res.status_code == 200 else None
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        # Fallback to local pipeline
        if local_pipeline and local_pipeline.user_features is not None:
            df = local_pipeline.user_features
            high_risk_df = df[df["Risk Band"] == "High"]
            total_loss = high_risk_df["Financial Exposure ($)"].sum()
            blocked_logs = [log for log in local_pipeline.logs 
                            if log.get("Action") == "Refund Blocked"]
            recovered_losses = 0.0
            blocked_user_ids = []
            for log in blocked_logs:
                for word in log.get("Detail", "").split():
                    if word.startswith("USER"):
                        blocked_user_ids.append(word.rstrip("."))
            if blocked_user_ids:
                blocked_df = df[df['user_id'].isin(blocked_user_ids)]
                recovered_losses = float(blocked_df["Financial Exposure ($)"].sum())
            return {
                "total_users": len(df),
                "high_risk_flagged": len(high_risk_df),
                "financial_exposure": float(total_loss),
                "recovered_losses": float(recovered_losses)
            }
        return None
    except Exception as e:
        st.error(f"Pipeline error: {str(e)}")
        return None

def fetch_users(query=""):
    try:
        res = requests.get(f"{config.API_BASE_URL}/users", params={"query": query}, timeout=config.API_TIMEOUT)
        if res.status_code == 200 and isinstance(res.json(), list):
            return pd.DataFrame(res.json())
        return pd.DataFrame()
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
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
    except (ValueError, KeyError):
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Pipeline error: {str(e)}")
        return pd.DataFrame()

def fetch_user_details(user_id):
    try:
        res = requests.get(f"{config.API_BASE_URL}/users/{user_id}", timeout=config.API_TIMEOUT)
        return res.json() if res.status_code == 200 else None
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        if local_pipeline:
            return local_pipeline.get_user_profile(user_id)
        return None
    except Exception as e:
        st.error(f"Pipeline error: {str(e)}")
        return None

def fetch_heatmap_data():
    try:
        res = requests.get(f"{config.API_BASE_URL}/heatmap-data", timeout=config.API_TIMEOUT)
        if res.status_code == 200 and isinstance(res.json(), list):
            return pd.DataFrame(res.json())
        return pd.DataFrame()
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        if local_pipeline and local_pipeline.user_features is not None:
            df = local_pipeline.user_features
            return df[['user_id', 'Return Frequency', 'High-Value Item Ratio',
                       'Avg Time-to-Return', 'Reason Diversity', 'Top Reason Ratio',
                       'Days Active', 'Risk Score', 'Risk Band']].head(500)
        return pd.DataFrame()
    except (ValueError, KeyError):
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Pipeline error: {str(e)}")
        return pd.DataFrame()

def fetch_logs():
    try:
        res = requests.get(f"{config.API_BASE_URL}/logs", timeout=config.API_TIMEOUT)
        if res.status_code == 200 and isinstance(res.json(), list):
            return pd.DataFrame(res.json())
        return pd.DataFrame()
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        if local_pipeline:
            return pd.DataFrame(local_pipeline.logs)
        return pd.DataFrame()
    except (ValueError, KeyError):
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Pipeline error: {str(e)}")
        return pd.DataFrame()

def fetch_pipeline_runs():
    try:
        res = requests.get(f"{config.API_BASE_URL}/pipeline-runs", timeout=config.API_TIMEOUT)
        if res.status_code == 200:
            return res.json()
        return []
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        try:
            from backend import database
            return database.get_pipeline_runs()
        except Exception:
            return []
    except Exception:
        return []

def fetch_all_users_for_chart():
    try:
        res = requests.get(f"{config.API_BASE_URL}/users/all-bands", 
                           timeout=config.API_TIMEOUT)
        if res.status_code == 200:
            return pd.DataFrame(res.json())
        return pd.DataFrame()
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        if local_pipeline and local_pipeline.user_features is not None:
            df = local_pipeline.user_features
            return df[['user_id', 'Risk Band']].copy()
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# --- SIDEBAR ---
st.sidebar.title("RevGuard")
st.sidebar.caption("Returns Fraud Detection Engine")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation", 
    ["Risk Overview", "User Investigation", "Behavioral Analytics", "Data Ingestion", "Audit Logs"]
)

st.sidebar.divider()
st.sidebar.markdown("### Engine Settings")
st.sidebar.info("Model thresholds are derived dynamically via Isolation Forest min-max scaling.")

stats = fetch_risk_stats()
is_standalone = False

# Check if actually connected to API or using fallback
try:
    api_check = requests.get(f"{config.API_BASE_URL.replace('/api/v1', '')}/", timeout=0.5)
    if api_check.status_code != 200:
        is_standalone = True
except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
    is_standalone = True

if not is_standalone:
    st.sidebar.success("Connected to API [Online]")
    st.sidebar.caption("Model Status: Trained [Ready]")
elif local_pipeline and local_pipeline.user_features is not None:
    st.sidebar.warning("Standalone Mode [No API]")
    if local_pipeline.trained_at:
        st.sidebar.caption(f"Model trained: {local_pipeline.trained_at}")
        try:
            from datetime import datetime
            trained_dt = datetime.strptime(local_pipeline.trained_at, "%Y-%m-%d %H:%M:%S")
            hours_since = (datetime.now() - trained_dt).total_seconds() / 3600
            if hours_since > 24:
                st.sidebar.warning(f"Model is stale ({hours_since:.0f}h old) — consider retraining")
            else:
                st.sidebar.success(f"Model is fresh ({hours_since:.1f}h old)")
        except Exception:
            pass
    source = "loaded from disk" if local_pipeline.model_loaded_from_disk else "trained this session"
    st.sidebar.caption(f"Source: {source}")
else:
    st.sidebar.error("No model loaded. Go to Data Ingestion.")

if stats:
    st.sidebar.divider()
    st.sidebar.metric("Total Users", f"{stats['total_users']:,}")
    st.sidebar.metric("High Risk", f"{stats['high_risk_flagged']:,}")


# ==========================================
# PAGE 1: RISK OVERVIEW
# ==========================================
if page == "Risk Overview":
    st.title("Enterprise Risk Overview")
    st.markdown("Monitor real-time e-commerce return fraud metrics, exposure distributions, and system health.")
    
    if not stats or "error" in stats:
        st.warning("Backend API is unavailable or model hasn't been initialized. Go to 'Data Ingestion'.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Total Active Users (Scored)", f"{stats['total_users']:,}")
        col2.metric("High Risk Users (Flagged)", f"{stats['high_risk_flagged']}")
        col3.metric("Fraud Financial Exposure", f"${stats['financial_exposure']:,.2f}", "-12% vs last month", delta_color="inverse")
        col4.metric("Losses Recovered (Blocked Refunds)", f"${stats['recovered_losses']:,.2f}")
        
        st.divider()
        
        all_users_df = fetch_all_users_for_chart()
        users_df = fetch_users()
        if not all_users_df.empty:
            col_chart1, col_chart2 = st.columns([1, 1])
            
            with col_chart1:
                st.subheader("Risk Band Distribution (All Users)")
                fig_pie = px.pie(
                    all_users_df, names='Risk Band', 
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
elif page == "User Investigation":
    st.title("User Risk Investigation")
    st.markdown("Search a specific user's behavioral patterns, SHAP explanation factors, and audit historical actions.")
    
    search_query = st.text_input("Enter User ID (USER00000001) or Order ID (ORD00000001)", placeholder="Enter User ID (USER00000001) or Order ID (ORD00000001)")
    
    # Task 5: Add input validation in search box
    if search_query:
        search_query = search_query.strip()
        if not (search_query.startswith('USER') or search_query.startswith('ORD')):
            st.warning("Please enter a valid User ID starting with USER or Order ID starting with ORD")
            st.stop()
    
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
                st.markdown(f"**Days Active:** {user_data.get('Days Active', 0)} days")
                st.markdown(f"**Reason Diversity:** {user_data.get('Reason Diversity', 0):.2f}")
                
                st.write("")
                st.divider()
                st.subheader("Investigator Actions")

                analyst_name = st.text_input(
                    "Your name (required to take any action)",
                    placeholder="e.g. Rahul Sharma",
                    key="analyst_name"
                )
                action_notes = st.text_area(
                    "Notes (optional)",
                    placeholder="Reason for this action...",
                    key="action_notes",
                    height=80
                )

                col_block, col_safe = st.columns(2)

                with col_block:
                    if st.button("Block Refund & Flag Account", type="primary", 
                                 use_container_width=True):
                        if local_pipeline:
                            local_pipeline.add_log(
                                "Refund Blocked", 
                                "Investigator",
                                f"User {user_data['User ID']} manually flagged. Score: {score}. Band: {band}."
                            )
                        st.error(f"🚫 Account {user_data['User ID']} has been flagged. Refund blocked.")
                        st.toast("Action logged to audit trail.", icon="🔒")

                with col_safe:
                    if st.button("Mark as Safe (False Positive)", use_container_width=True):
                        if local_pipeline:
                            local_pipeline.add_log(
                                "Marked Safe", 
                                "Investigator",
                                f"User {user_data['User ID']} marked as false positive. Score was: {score}."
                            )
                        st.success(f"✅ User {user_data['User ID']} marked as safe.")
                        st.toast("Action logged to audit trail.", icon="✅")
                
            with col_shap:
                st.subheader("Why was this user flagged? (SHAP Explainability)")
                
                if 'SHAP' in user_data:
                    shap_df = pd.DataFrame(user_data['SHAP'])
                    # Task 4: Visualization improvements
                    # Use Abs_Contribution for bar size, Direction for color
                    # Increases risk = Red (#FF4B4B), Decreases risk = Green (#00CC96)
                    
                    # Sort bars by Abs_Contribution descending
                    shap_df = shap_df.sort_values(by="Abs_Contribution", ascending=True) # Ascending true because Plotly displays from bottom up
                    
                    if not shap_df.empty:
                        fig_shap = px.bar(
                            shap_df, 
                            x="Abs_Contribution", 
                            y="Feature", 
                            orientation='h',
                            color="Direction",
                            color_discrete_map={
                                'increases_risk': '#FF4B4B',
                                'decreases_risk': '#00CC96'
                            },
                            title="Feature Impact (Magnitude & Direction)"
                        )
                        fig_shap.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
                        st.plotly_chart(fig_shap, use_container_width=True)
                        st.caption("**Legend:** Red = pushes risk higher | Green = pushes risk lower")
                    else:
                        st.info("No significant feature contributions found.")
                else:
                    st.info("SHAP data unavailable for this user.")
                
                # RAG Chatbot Section for User Investigation
                st.subheader("Ask about this user")
                
                current_user_id = user_data.get("User ID", "")
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = {}
                if current_user_id not in st.session_state.chat_history:
                    st.session_state.chat_history[current_user_id] = []

                # Render prior chat history for this user
                for msg in st.session_state.chat_history[current_user_id]:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])
                        if msg["role"] == "assistant" and "sources_used" in msg:
                            st.caption(f"Grounded in {msg['sources_used']} data points")

                # Capture new question
                user_prompt = st.chat_input("Ask why this user is flagged, e.g. 'why is the return frequency so high?'")
                if user_prompt:
                    st.session_state.chat_history[current_user_id].append({"role": "user", "content": user_prompt})
                    with st.chat_message("user"):
                        st.write(user_prompt)

                    timeout_val = max(config.API_TIMEOUT, 15)
                    with st.spinner("Analyzing user case context..."):
                        try:
                            res = requests.post(
                                f"{config.API_BASE_URL}/users/{current_user_id}/chat",
                                json={"question": user_prompt},
                                timeout=timeout_val
                            )
                            if res.status_code == 200:
                                data = res.json()
                                answer = data.get("answer", "No answer received.")
                                sources_used = data.get("sources_used", 0)
                                st.session_state.chat_history[current_user_id].append({
                                    "role": "assistant",
                                    "content": answer,
                                    "sources_used": sources_used
                                })
                                with st.chat_message("assistant"):
                                    st.write(answer)
                                    st.caption(f"Grounded in {sources_used} data points")
                            elif res.status_code == 503:
                                st.error("Model not initialized. Please run the pipeline first.")
                            else:
                                st.error(f"Chat API error ({res.status_code}): {res.text}")
                        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                            if local_pipeline:
                                try:
                                    from backend.rag import service as local_rag_service
                                    data = local_rag_service.answer_question(current_user_id, user_prompt, local_pipeline)
                                    answer = data.get("answer", "No answer received.")
                                    sources_used = data.get("sources_used", 0)
                                    st.session_state.chat_history[current_user_id].append({
                                        "role": "assistant",
                                        "content": answer,
                                        "sources_used": sources_used
                                    })
                                    with st.chat_message("assistant"):
                                        st.write(answer)
                                        st.caption(f"Grounded in {sources_used} data points")
                                except Exception as e:
                                    st.error(f"Chat error: {str(e)}")
                            else:
                                st.error("Unable to connect to backend API server.")
                        except Exception as e:
                            st.error(f"Chat error: {str(e)}")
                
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
elif page == "Behavioral Analytics":
    st.title("Behavioral Analytics")
    st.markdown("Comparison of average return behaviors across Risk Bands.")
    
    heatmap_df = fetch_heatmap_data()
    if not heatmap_df.empty:
        st.subheader("Behavioral Patterns by Risk Band")
        
        # Calculate averages per risk band for a cleaner comparison
        comparison_df = heatmap_df.groupby('Risk Band')[['Return Frequency', 'High-Value Item Ratio', 'Risk Score']].mean().reset_index()
        
        # Reshape data for plotting
        plot_df = comparison_df.melt(id_vars='Risk Band', var_name='Metric', value_name='Average Value %')
        # Filter for the two main metrics
        all_metrics = ['Return Frequency', 'High-Value Item Ratio', 
                       'Avg Time-to-Return', 'Reason Diversity', 
                       'Top Reason Ratio', 'Days Active']
        available_metrics = [m for m in all_metrics if m in comparison_df.columns]
        selected_metrics = st.multiselect(
            "Select features to compare", 
            options=available_metrics, 
            default=['Return Frequency', 'High-Value Item Ratio']
        )
        plot_df = plot_df[plot_df['Metric'].isin(selected_metrics)]
        
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
        st.info("Insight: High Risk users typically show significantly higher Return Frequency and High-Value Item Ratios compared to Low Risk users.")
    else:
        st.warning("No data available.")

# ==========================================
# PAGE 4: DATA INGESTION
# ==========================================
elif page == "Data Ingestion":
    st.title("Data Ingestion Pipeline")
    st.markdown("Upload raw transaction and return logs to execute the ML pipeline.")
    
    uploaded_file = st.file_uploader("Upload CSV transaction data", type=["csv"])

    if uploaded_file is not None:
        with st.spinner("Saving uploaded file and retraining model..."):
            try:
                import io
                new_df = pd.read_csv(uploaded_file)
                required_cols = ['user_id', 'order_id', 'purchase_date', 
                                 'return_date', 'return_reason', 'refund_amount']
                missing = [c for c in required_cols if c not in new_df.columns]
                if missing:
                    st.error(f"Uploaded CSV is missing required columns: {missing}")
                else:
                    os.makedirs(os.path.dirname(config.DATA_PATH), exist_ok=True)
                    new_df.to_csv(config.DATA_PATH, index=False)
                    if local_pipeline:
                        local_pipeline.run()
                        st.success(f"✅ File uploaded ({len(new_df):,} rows). Model retrained.")
                        st.rerun()
            except Exception as e:
                st.error(f"Upload failed: {str(e)}")

    st.divider()
    if st.button("Trigger Pipeline Manually (Use existing CSV)"):
        with st.spinner("Retraining on existing data..."):
            try:
                if not is_standalone:
                    res = requests.post(f"{config.API_BASE_URL}/run-pipeline", timeout=5)
                    if res.status_code == 200:
                        st.success("✅ Pipeline triggered via API.")
                        st.info("⏳ Pipeline is retraining in the background. "
                                "Refresh the Risk Overview page in 10-15 seconds "
                                "to see updated scores.")
                    else:
                        st.error("API pipeline trigger failed.")
                else:
                    if local_pipeline:
                        local_pipeline.run()
                        st.success("✅ Local pipeline retrained successfully.")
                        st.rerun()
                    else:
                        st.error("No local pipeline available.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    if st.button("🔄 Refresh Dashboard Data"):
        st.cache_resource.clear()
        st.rerun()
    
    st.divider()
    st.subheader("Force Retrain")
    st.caption("Retrain the model from scratch, ignoring any saved model on disk. Use when new data has been added.")
    if st.button("Force Full Retrain (ignore saved model)"):
        with st.spinner("Retraining from scratch — this takes 30-60 seconds..."):
            try:
                if local_pipeline:
                    local_pipeline.run()
                    st.success("Retraining complete. All scores updated.")
                    st.rerun()
            except Exception as e:
                st.error(f"Retraining failed: {str(e)}")

# ==========================================
# PAGE 5: AUDIT LOGS
# ==========================================
elif page == "Audit Logs":
    st.title("System Audit Logs")
    st.markdown("Immutable record of automated system events and investigator actions for compliance.")
    
    logs_df = fetch_logs()
    if not logs_df.empty:
        st.caption(f"{len(logs_df)} total log entries")
        st.dataframe(logs_df, use_container_width=True, hide_index=True)
    else:
        st.info("No system logs currently recorded.")
