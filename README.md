# signal_quality_predictor
# Intelligent Signal Quality Predictor (GSM / 4G / 5G)

An AI-powered system that predicts the quality of a mobile network signal (GSM, 4G, 5G) using technical signal features. This project includes synthetic data generation, machine learning training, and a real-time prediction interface using Streamlit.

--

## Project Overview

This tool allows users to:
- Input mobile signal metrics (RSRP, RSRQ, SINR, RSSI, etc.)
- Predict signal quality (Excellent, Good, Fair, Poor)
- Visualize results instantly with a user-friendly web interface

It’s useful for telecom simulations, signal analysis, and educational demonstrations.


---

##  Features Used for Prediction

| Feature                  | Description                                |
|--------------------------|--------------------------------------------|
| `RSRP_dBm`              | Reference signal received power            |
| `RSRQ_dB`               | Reference signal received quality          |
| `SINR_dB`               | Signal-to-interference-plus-noise ratio    |
| `RSSI_dBm`              | Received signal strength indicator         |
| `Data_Throughput_Mbps` | Data speed in Mbps                          |
| `Latitude`              | Location latitude                          |
| `Longitude`             | Location longitude                         |
| `Network_Type`          | GSM / 4G / 5G                              |

The target feature is `Signal_Quality`: one of **Excellent, Good, Fair, Poor**.

---

##  Model Training

The model is trained in `training.ipynb` using a **RandomForestClassifier**. The training uses synthetic data generated via `synthetic_data.py`.

Saved models and encoders:
- `rf_model.joblib` – Trained model
- `network_encoder.joblib` – Encodes network type (GSM=0, 4G=1, 5G=2)
- `target_encoder.joblib` – Encodes target signal quality
- `scaler.joblib` – Scales input features

---

##  Running the Streamlit App

 1. Clone the Repository

```bash
git clone https://github.com/Wajihx/signal_quality_predictor.git
cd signal_quality_predictor
```
2. Install Requirements
```bash
pip install -r requirements.txt
```
3. Launch the Interface
```bash
streamlit run main.py
```

