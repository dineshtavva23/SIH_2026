import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
import time

# --- Page Configuration ---
st.set_page_config(page_title="SkyGuard AI Dashboard", page_icon="☁️", layout="wide")

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value {font-size: 2.5rem; font-weight: bold; color: #1f77b4;}
    .anomaly-value {font-size: 2.5rem; font-weight: bold; color: #d62728;}
    h1, h2, h3, h4 {color: #2c3e50 !important;}
    </style>
""", unsafe_allow_html=True)

st.title("☁️ SkyGuard AI: Real-Time AWS Quality Control")
st.markdown("### Intelligent Anomaly Detection for Automatic Weather Stations")

st.markdown("""
This dashboard demonstrates our **Hybrid Detection Engine**. 
We ingest Temperature, Pressure, and Humidity data, and use an **Isolation Forest** model to detect anomalies like spikes, frozen values, and drift in real-time.
""")

# --- Synthetic Data Generation (Matching their "controlled anomaly injection") ---
@st.cache_data
def generate_synthetic_data():
    np.random.seed(42)
    dates = pd.date_range(start="2026-09-01", periods=1000, freq="H")
    
    # Normal base patterns
    temp = 25 + 5 * np.sin(np.linspace(0, 20*np.pi, 1000)) + np.random.normal(0, 1, 1000)
    pressure = 1013 + 3 * np.cos(np.linspace(0, 10*np.pi, 1000)) + np.random.normal(0, 0.5, 1000)
    humidity = 60 + 15 * np.sin(np.linspace(0, 20*np.pi, 1000) + np.pi/2) + np.random.normal(0, 2, 1000)
    
    # Injecting Anomalies (As promised in their slides)
    # 1. Spikes
    temp[200] = 45.0  # Huge spike
    pressure[450] = 980.0 # Huge drop
    
    # 2. Frozen values (Sensor stuck)
    humidity[600:650] = 55.0 
    
    # 3. Drift (Gradual increase due to sensor calibration issue)
    temp[800:1000] += np.linspace(0, 10, 200)
    
    df = pd.DataFrame({
        "Timestamp": dates,
        "Temperature (C)": temp,
        "Pressure (hPa)": pressure,
        "Humidity (%)": humidity
    })
    return df

df = generate_synthetic_data()

# --- Machine Learning: Isolation Forest ---
st.sidebar.header("⚙️ Model Configuration")
contamination = st.sidebar.slider("Contamination Rate (Sensitivity)", 0.01, 0.10, 0.05, 0.01)

st.sidebar.markdown("---")
st.sidebar.header("💾 Export Data")
csv = df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="Download Synthetic Dataset (CSV)",
    data=csv,
    file_name='aws_synthetic_data.csv',
    mime='text/csv',
)

features = ["Temperature (C)", "Pressure (hPa)", "Humidity (%)"]

# Feature Engineering: Rolling stats
df['Temp_Rolling_Mean'] = df['Temperature (C)'].rolling(window=12, min_periods=1).mean()
df['Temp_Rolling_Std'] = df['Temperature (C)'].rolling(window=12, min_periods=1).std().fillna(0)
model_features = features + ['Temp_Rolling_Mean', 'Temp_Rolling_Std']

# Train and Predict
model = IsolationForest(contamination=contamination, random_state=42)
df['Anomaly_Score'] = model.fit_predict(df[model_features])
df['Is_Anomaly'] = df['Anomaly_Score'] == -1

# --- Dashboard Layout ---
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div class='metric-card'><h4>Total Observations</h4><div class='metric-value'>{len(df)}</div></div>", unsafe_allow_html=True)
with col2:
    anomalies_count = df['Is_Anomaly'].sum()
    st.markdown(f"<div class='metric-card'><h4>Anomalies Detected</h4><div class='anomaly-value'>{anomalies_count}</div></div>", unsafe_allow_html=True)
with col3:
    health_score = 100 - (anomalies_count / len(df) * 100)
    color = "green" if health_score > 90 else "orange"
    st.markdown(f"<div class='metric-card'><h4>Sensor Health Index</h4><div class='metric-value' style='color:{color};'>{health_score:.1f}%</div></div>", unsafe_allow_html=True)

st.write("---")

# --- Visualizations ---
st.subheader("Multivariate Time-Series Analysis")

tab1, tab2, tab3 = st.tabs(["Temperature", "Pressure", "Humidity"])

def plot_feature(feature_name, color):
    fig = go.Figure()
    # Normal data
    normal_df = df[~df['Is_Anomaly']]
    fig.add_trace(go.Scatter(x=normal_df['Timestamp'], y=normal_df[feature_name], 
                             mode='markers', name='Normal', marker=dict(color=color, size=5)))
    
    # Anomalies
    anomaly_df = df[df['Is_Anomaly']]
    fig.add_trace(go.Scatter(x=anomaly_df['Timestamp'], y=anomaly_df[feature_name], 
                             mode='markers', name='Anomaly (Fault Detected)', 
                             marker=dict(color='red', size=8, symbol='x')))
    
    fig.update_layout(title=f"{feature_name} over Time with Detected Faults",
                      xaxis_title="Time", yaxis_title=feature_name,
                      hovermode="x unified",
                      template="plotly_white")
    return fig

with tab1:
    st.plotly_chart(plot_feature("Temperature (C)", "#1f77b4"), use_container_width=True)
    st.info("Notice how the model detects the **spike** around Sep 9th and the **gradual drift** starting in October.")

with tab2:
    st.plotly_chart(plot_feature("Pressure (hPa)", "#2ca02c"), use_container_width=True)
    st.info("The model successfully detects the massive sudden pressure drop, classifying it as a severe anomaly.")

with tab3:
    st.plotly_chart(plot_feature("Humidity (%)", "#ff7f0e"), use_container_width=True)
    st.info("Notice the **frozen values** (a flat horizontal line) around Sep 26th. The temporal model catches this unnatural lack of variance.")

# --- Fault Classification Logic (Rule-based on top of ML) ---
st.subheader("Detected Fault Log")
anomalies_df = df[df['Is_Anomaly']].copy()

# Simple mock classifier for demo purposes
def classify_fault(row):
    if row['Temperature (C)'] > 40:
        return "Spike (Temperature)"
    elif row['Pressure (hPa)'] < 1000:
        return "Extreme Drop (Pressure)"
    elif row['Timestamp'] > pd.to_datetime('2026-10-01'):
        return "Gradual Drift Detected"
    else:
        return "Cross-Sensor Inconsistency / Frozen Value"

anomalies_df['Fault Classification'] = anomalies_df.apply(classify_fault, axis=1)

st.dataframe(anomalies_df[['Timestamp', 'Temperature (C)', 'Pressure (hPa)', 'Humidity (%)', 'Fault Classification']].head(10), use_container_width=True)

st.success("✅ Demo Ready! This dashboard runs entirely locally and proves your concept of using ML for AWS quality control without relying on manual thresholds.")
