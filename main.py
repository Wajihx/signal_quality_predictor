import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load saved model and encoders once
@st.cache_resource
def load_model_files():
    model = joblib.load('rf_model.joblib')
    network_encoder = joblib.load('network_encoder.joblib')
    target_encoder = joblib.load('target_encoder.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, network_encoder, target_encoder, scaler

model, network_encoder, target_encoder, scaler = load_model_files()

st.title("Intelligent GSM/4G/5G Signal Quality Predictor 📶")

st.markdown("""
Enter the characteristics of your signal below to predict its quality.
""")

# User inputs
rsrp = st.slider("RSRP (dBm)", min_value=-150, max_value=-50, value=-95)
rsrq = st.slider("RSRQ (dB)", min_value=-50, max_value=-0, value=-10)
sinr = st.slider("SINR (dB)", min_value=-30, max_value=30, value=10)
rssi = st.slider("RSSI (dBm)", min_value=-100, max_value=-50, value=-75)
throughput = st.slider("Data Throughput (Mbps)", min_value=0.0, max_value=300.0, value=30.0, step=0.1)
latitude = st.number_input("Latitude", min_value=10.0, max_value=50.0, value=35.0, format="%.6f")
longitude = st.number_input("Longitude", min_value=-20.0, max_value=20.0, value=5.0, format="%.6f")
network_type = st.selectbox("Network Type", options=network_encoder.classes_)

# Prepare input for prediction
input_dict = {
    'RSRP_dBm': rsrp,
    'RSRQ_dB': rsrq,
    'SINR_dB': sinr,
    'RSSI_dBm': rssi,
    'Data_Throughput_Mbps': throughput,
    'Latitude': latitude,
    'Longitude': longitude,
    'Network_Type_Encoded': network_encoder.transform([network_type])[0]
}

input_df = pd.DataFrame([input_dict])
input_scaled = scaler.transform(input_df)

# Predict button
if st.button("Predict Signal Quality"):
    pred_encoded = model.predict(input_scaled)
    pred_label = target_encoder.inverse_transform(pred_encoded)[0]
    
    st.success(f"**Predicted Signal Quality:** {pred_label}")

    # Optional: add some explanation or tips based on quality
    if pred_label == "Excellent":
        st.balloons()
        st.write("Your signal quality is excellent. You should experience very reliable connectivity.")
    elif pred_label == "Good":
        st.write("Good signal quality. Connection should be stable with minor issues.")
    elif pred_label == "Fair":
        st.warning("Fair quality. You might experience some connection drops or slowdowns.")
    else:
        st.error("Poor signal quality. Consider moving to a better location or checking your network.")


