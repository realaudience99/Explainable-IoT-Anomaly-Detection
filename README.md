# Explainable IoT Anomaly Detection using Autoencoders

This project focuses on detecting cyber attacks in IoT networks using Deep Learning.

## Current Progress (Week 4-5)
* **Dataset:** CIC-IoT2023 (1.1M+ records)
* **Preprocessing:** 39 features reduced to 26 key features using correlation and variance analysis.
* **Model:** Trained Autoencoder architecture with 9-neuron bottleneck layer.
* **Environment:** Kali Linux with TensorFlow/Keras on i7-12700H & RTX 4060.

## Repository Structure
* `models/`: Contains the trained `.keras` model.
* `config/`: Contains the `final_features.gz` list for feature consistency.
* `scripts/`: Contains the `analyst.py` for data processing and inference.

---
*Note: Explainable AI (XAI) integration using SHAP is currently in progress.*
