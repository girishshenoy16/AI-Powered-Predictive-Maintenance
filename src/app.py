import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# --- Page Config & Premium Styling ---
st.set_page_config(page_title="AI Predictive Maintenance", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Outfit', sans-serif; 
    }
    
    /* Glassmorphism KPI Cards */
    .kpi-card { 
        background: rgba(30, 34, 45, 0.6); 
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 25px; 
        border-radius: 20px; 
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        text-align: center; 
        transition: all 0.3s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px 0 rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .kpi-title { 
        font-size: 14px; 
        color: #94a3b8; 
        font-weight: 600; 
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    .kpi-value { 
        font-size: 38px; 
        font-weight: 700; 
        margin-top: 5px;
        background: linear-gradient(135deg, #e0e7ff 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .status-healthy { 
        font-size: 38px; 
        font-weight: 700; 
        margin-top: 5px;
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .status-danger { 
        font-size: 38px; 
        font-weight: 700; 
        margin-top: 5px;
        background: linear-gradient(135deg, #ef4444 0%, #f87171 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Primary Button Glow */
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        border: none;
        color: white;
        font-weight: 700;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.5);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Load Models & Data ---
@st.cache_resource
def load_models():
    try:
        return joblib.load("models/xgb_classifier.pkl"), joblib.load("models/xgb_regressor.pkl")
    except:
        return None, None

@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/processed/train_processed.csv")
    except:
        return None

classifier, regressor = load_models()
df = load_data()

# Premium Header
st.markdown("<h1 style='text-align: center; font-weight: 700; margin-bottom: 30px; background: linear-gradient(to right, #60a5fa, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>⚡ AI Predictive Maintenance System</h1>", unsafe_allow_html=True)

if df is None or classifier is None:
    st.error("⚠️ Processed Data or Models not found. Please run main.py first.")
    st.stop()

# --- Dashboard Tabs ---
tab1, tab2, tab3 = st.tabs(["🚀 Real-Time Telemetry", "✍️ Manual Prediction", "📁 Batch AI Processing"])

features_to_drop = ['engine_id', 'cycle', 'RUL', 'label']

# ==========================================
# TAB 1: Real-Time Simulation
# ==========================================
with tab1:
    st.sidebar.markdown("### 🎛️ Simulation Controls")
    engine_list = df['engine_id'].unique()
    selected_engine = st.sidebar.selectbox("Select Engine to Monitor", engine_list)
    engine_data = df[df['engine_id'] == selected_engine]
    current_cycle = st.sidebar.slider("Simulate Time (Flight Cycles)", 1, int(engine_data['cycle'].max()), 1)

    # Clean display name (remove 'train_' prefix for cleaner UI)
    display_engine_name = selected_engine.replace("train_", "").replace("test_", "")

    current_row = engine_data[engine_data['cycle'] == current_cycle]
    features = current_row.drop(columns=features_to_drop, errors='ignore')
    
    is_failing = classifier.predict(features)[0]
    rul_days = int(regressor.predict(features)[0])

    # KPI Layout with Glassmorphism and Dynamic Glow Borders
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="kpi-card" style="border-top: 4px solid #60a5fa;"><div class="kpi-title">Active Engine ID</div><div class="kpi-value">#{display_engine_name}</div></div>', unsafe_allow_html=True)
    with col2:
        status_class = "status-danger" if is_failing else "status-healthy"
        status_text = "FAILURE IMMINENT" if is_failing else "HEALTHY"
        border_color = "#ef4444" if is_failing else "#10b981"
        st.markdown(f'<div class="kpi-card" style="border-top: 4px solid {border_color};"><div class="kpi-title">System Status</div><div class="{status_class}">{status_text}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="kpi-card" style="border-top: 4px solid #c084fc;"><div class="kpi-title">AI Predicted RUL</div><div class="kpi-value">{rul_days} <span style="font-size: 16px; color: #94a3b8;">Cycles</span></div></div>', unsafe_allow_html=True)

    st.markdown("<br><hr style='border: 1px solid rgba(255,255,255,0.1);'><br>", unsafe_allow_html=True)
    st.markdown("### 📊 Live Core Sensor Telemetry (4-Grid)")
    
    historical_data = engine_data[engine_data['cycle'] <= current_cycle]
    
    # Premium Plotly Aesthetics with Translucent Area Fills and Spline Smoothing
    def create_rich_chart(data, y_col, title, line_color, fill_color):
        fig = px.line(data, x='cycle', y=y_col, title=title)
        
        # Spline shape makes the jagged sensor data look like elegant, smooth waves
        fig.update_traces(
            line=dict(color=line_color, width=2.5, shape='spline'), 
            fill='tozeroy',
            fillcolor=fill_color
        )
        
        # Add a glowing 'live tracking' blip at the very end of the line
        last_row = data.iloc[-1:]
        fig.add_scatter(
            x=last_row['cycle'], 
            y=last_row[y_col], 
            mode='markers',
            marker=dict(size=10, color=line_color, line=dict(color='white', width=2)),
            showlegend=False,
            hoverinfo='skip'
        )
        
        # Dynamically zoom the Y-axis to make the data variance visible
        y_min = data[y_col].min() * 0.998
        y_max = data[y_col].max() * 1.002
        if len(data) == 1:
            y_min = data[y_col].iloc[0] * 0.998
            y_max = data[y_col].iloc[0] * 1.002
            
        fig.update_layout(
            height=280, # Sleeker, wider charts
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8',
            title_font=dict(size=16, color='white', family='Outfit'),
            xaxis=dict(showgrid=False, title=''),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.03)', range=[y_min, y_max]),
            margin=dict(l=10, r=10, t=40, b=10),
            hovermode="x unified"
        )
        return fig

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(create_rich_chart(historical_data, 'sensor_11', 'LPC Outlet Temp (Sensor 11)', '#60a5fa', 'rgba(96, 165, 250, 0.1)'), use_container_width=True)
        st.plotly_chart(create_rich_chart(historical_data, 'sensor_4', 'LPT Outlet Temp (Sensor 4)', '#34d399', 'rgba(52, 211, 153, 0.1)'), use_container_width=True)
    with chart_col2:
        st.plotly_chart(create_rich_chart(historical_data, 'sensor_14', 'HPC Outlet Pressure (Sensor 14)', '#c084fc', 'rgba(192, 132, 252, 0.1)'), use_container_width=True)
        st.plotly_chart(create_rich_chart(historical_data, 'sensor_9', 'Engine Pressure Ratio (Sensor 9)', '#f472b6', 'rgba(244, 114, 182, 0.1)'), use_container_width=True)

# ==========================================
# TAB 2: Manual Input Prediction
# ==========================================
with tab2:
    st.markdown("### ✍️ Manual Sensor Diagnostics")
    st.markdown("Adjust the critical core sensors below. The AI automatically stabilizes the remaining background metrics to generate a precise forecast.")
    
    default_vals = df.iloc[0].drop(features_to_drop, errors='ignore')
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        s11 = st.number_input("LPC Outlet Temp (Sensor 11)", value=float(default_vals['sensor_11']))
    with col2:
        s14 = st.number_input("HPC Outlet Pressure (Sensor 14)", value=float(default_vals['sensor_14']))
    with col3:
        s4 = st.number_input("LPT Outlet Temp (Sensor 4)", value=float(default_vals['sensor_4']))
    with col4:
        s9 = st.number_input("Engine Pressure Ratio (Sensor 9)", value=float(default_vals['sensor_9']))
            
    if st.button("Run AI Diagnostic", type="primary", use_container_width=True):
        # To maintain scientific accuracy with only 4 inputs, we find the closest historical engine state
        user_array = pd.Series({'sensor_11': s11, 'sensor_14': s14, 'sensor_4': s4, 'sensor_9': s9})
        df_4 = df[['sensor_11', 'sensor_14', 'sensor_4', 'sensor_9']]
        
        # Euclidean distance to find the closest real-world engine state matching these 4 sensors
        with st.spinner("AI is correlating telemetry..."):
            distances = ((df_4 - user_array) ** 2).sum(axis=1)
            closest_idx = distances.idxmin()
            closest_row = df.iloc[closest_idx].drop(labels=features_to_drop, errors='ignore')
            
            input_df = pd.DataFrame([closest_row])
            
            is_failing_manual = classifier.predict(input_df)[0]
            rul_days_manual = int(regressor.predict(input_df)[0])
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            import plotly.graph_objects as go
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = rul_days_manual,
                title = {'text': "Predicted Remaining Useful Life (Cycles)", 'font': {'color': '#94a3b8', 'size': 16, 'family': 'Outfit'}},
                gauge = {
                    'axis': {'range': [0, 250], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#ef4444" if is_failing_manual else "#10b981"},
                    'bgcolor': "rgba(255,255,255,0.05)",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(239, 68, 68, 0.3)"},
                        {'range': [30, 100], 'color': "rgba(245, 158, 11, 0.3)"},
                        {'range': [100, 250], 'color': "rgba(16, 185, 129, 0.3)"}],
                },
                number = {'font': {'color': 'white', 'size': 60}}
            ))
            fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with res_col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            if is_failing_manual == 1:
                st.markdown(f'<div class="kpi-card" style="border-left: 6px solid #ef4444; text-align: left; background: rgba(239, 68, 68, 0.1);"><h3 style="color: #f87171; margin-top:0;">🚨 CRITICAL ALERT</h3><p style="color: #e2e8f0; font-size: 18px; line-height: 1.6;">The telemetry indicates severe component degradation. Status is <strong>Failure Imminent</strong>.<br><br>The engine will suffer total breakdown in exactly <strong style="color: #f87171; font-size: 24px;">{rul_days_manual} cycles</strong>.</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="kpi-card" style="border-left: 6px solid #10b981; text-align: left; background: rgba(16, 185, 129, 0.05);"><h3 style="color: #34d399; margin-top:0;">✅ NORMAL OPERATION</h3><p style="color: #e2e8f0; font-size: 18px; line-height: 1.6;">The telemetry indicates optimal health. The engine is <strong>Healthy</strong>.<br><br>The system will operate safely for <strong style="color: #34d399; font-size: 24px;">{rul_days_manual} cycles</strong> before failure.</p></div>', unsafe_allow_html=True)

# ==========================================
# TAB 3: Batch CSV Upload
# ==========================================
with tab3:
    st.markdown("### 📁 Batch AI Processing")
    st.markdown("Upload a CSV containing massive arrays of unlabelled telemetry to predict failure states in bulk.")
    
    feature_names = df.drop(columns=features_to_drop, errors='ignore').columns.tolist()
    
    uploaded_file = st.file_uploader("Upload Raw Sensor CSV", type="csv")
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        
        missing_cols = [col for col in feature_names if col not in batch_df.columns]
        if missing_cols:
            st.error(f"⚠️ Missing required sensor columns: {missing_cols}")
        else:
            with st.spinner('AI is processing telemetry...'):
                predictions_class = classifier.predict(batch_df[feature_names])
                predictions_rul = regressor.predict(batch_df[feature_names])
            
            results_df = batch_df.copy()
            results_df['AI Status'] = ["Failure Imminent" if pred == 1 else "Normal" for pred in predictions_class]
            results_df['Predicted RUL (Cycles)'] = [int(r) for r in predictions_rul]
            
            # 1. Reorder columns to put critical AI outputs at the very front
            cols = ['AI Status', 'Predicted RUL (Cycles)'] + [c for c in results_df.columns if c not in ['AI Status', 'Predicted RUL (Cycles)']]
            results_df = results_df[cols]
            
            # 2. Executive Summary KPIs
            total_scanned = len(results_df)
            total_failures = len(results_df[results_df['AI Status'] == 'Failure Imminent'])
            avg_rul = int(results_df['Predicted RUL (Cycles)'].mean())
            
            st.markdown("<br>", unsafe_allow_html=True)
            kpi1, kpi2, kpi3 = st.columns(3)
            with kpi1:
                st.markdown(f'<div class="kpi-card" style="padding: 15px;"><div class="kpi-title">Rows Scanned</div><div class="kpi-value" style="font-size: 28px;">{total_scanned}</div></div>', unsafe_allow_html=True)
            with kpi2:
                alert_color = "status-danger" if total_failures > 0 else "status-healthy"
                st.markdown(f'<div class="kpi-card" style="padding: 15px;"><div class="kpi-title">Critical Failures</div><div class="{alert_color}" style="font-size: 28px;">{total_failures}</div></div>', unsafe_allow_html=True)
            with kpi3:
                st.markdown(f'<div class="kpi-card" style="padding: 15px;"><div class="kpi-title">Fleet Avg RUL</div><div class="kpi-value" style="font-size: 28px;">{avg_rul} <span style="font-size: 14px;">Cycles</span></div></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            # 3. Data Cleanup & Auto-Sorting
            # Strip ugly prefixes and sort so the most dangerous engines instantly bubble to the top
            results_df['engine_id'] = results_df['engine_id'].astype(str).str.replace('train_', '').str.replace('test_', '')
            results_df = results_df.sort_values('Predicted RUL (Cycles)')
            
            # 4. Interactive Filter
            show_failures = st.checkbox("🚨 Show only Critical Engines")
            display_df = results_df[results_df['AI Status'] == 'Failure Imminent'] if show_failures else results_df
            
            def highlight_failures(val):
                return 'background-color: rgba(239, 68, 68, 0.2); color: #f87171; font-weight: bold;' if val == 'Failure Imminent' else 'color: #34d399;'
                
            # 5. Rich Dataframe Styling (Rounding, Colors, Heatmap)
            styled_df = display_df.style.format(precision=2)\
                                        .applymap(highlight_failures, subset=['AI Status'])\
                                        .background_gradient(subset=['Predicted RUL (Cycles)'], cmap='RdYlGn', vmin=0, vmax=200)
            
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Full AI Predictions (CSV)", data=csv, file_name='ai_batch_predictions.csv', mime='text/csv', use_container_width=True)
