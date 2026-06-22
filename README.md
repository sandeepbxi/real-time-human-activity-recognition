# Human Activity Recognition using CNN-LSTM

## Overview

This project implements a real-time Human Activity Recognition (HAR) system using a hybrid CNN-LSTM deep learning architecture. The model classifies human activities from accelerometer and gyroscope sensor data and supports live predictions through a Streamlit web application.

The project was developed to explore deep learning for time-series sensor data and combines CNN-LSTM modeling, SHAP explainability, and live sensor streaming through PhyPhox to perform real-time activity classification.

## Project Images

Project screenshots, model visualizations, and deployment demonstrations can be found in the `images/` directory.

---

## Features

* Human Activity Recognition using Deep Learning
* CNN-LSTM Hybrid Architecture
* Real-Time Activity Prediction
* Streamlit-Based Interactive Dashboard
* SHAP Explainability
* Hyperparameter Tuning
* Live Sensor Streaming via PhyPhox
* TensorFlow/Keras Deployment

---

## Problem Statement

Human-machine interaction systems are becoming increasingly important in industrial automation, wearable devices, and smart environments.

The objective of this project is to accurately classify human activities using motion sensor data and provide real-time predictions through a deep learning model capable of learning both spatial and temporal patterns.

Potential applications include:

* Industrial Automation
* Worker Safety Monitoring
* Wearable Computing
* Smart Manufacturing
* Activity Tracking Systems

---

## Dataset

This project uses the **UCI Human Activity Recognition (UCI-HAR) Dataset**.

The dataset contains smartphone accelerometer and gyroscope readings collected from 30 participants performing six daily activities.

### Activities

* Walking
* Walking Upstairs
* Walking Downstairs
* Sitting
* Standing
* Laying

### Dataset Characteristics

* Sensors: Accelerometer and Gyroscope
* Sampling Rate: 50 Hz
* Window Size: 128 Samples
* Window Length: 2.56 Seconds
* Total Samples: ~10,299

The UCI-HAR dataset is already segmented and preprocessed, making it suitable for direct model training and evaluation.
Dataset Link: https://www.kaggle.com/datasets/drsaeedmohsen/ucihar-dataset

---

## Data Preparation

Since the dataset is already preprocessed, only the following preparation steps were required:

* Loading sensor signal files
* Reshaping data for CNN-LSTM input
* One-hot encoding activity labels
* Train-test splitting
* Formatting data for TensorFlow/Keras training

---

## Model Architecture

The proposed model combines Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks.

### Architecture Flow

```text
Input Sensor Data
        ↓
      Conv1D
        ↓
Batch Normalization
        ↓
    Max Pooling
        ↓
       LSTM
        ↓
      Dropout
        ↓
    Dense Layer
        ↓
Activity Prediction
```

### Why CNN + LSTM?

#### CNN

* Extracts local motion patterns from sensor signals
* Learns spatial relationships within the data
* Helps reduce the impact of noisy sensor readings

#### LSTM

* Captures temporal dependencies
* Learns sequential motion behavior
* Models long-term activity patterns

The hybrid architecture leverages the strengths of both approaches, resulting in robust activity recognition performance.

---

## Training Configuration

| Parameter               | Value                      |
| ----------------------- | -------------------------- |
| Optimizer               | Adam                       |
| Learning Rate           | 0.001                      |
| Batch Size              | 64                         |
| Epochs                  | 60                         |
| Loss Function           | Categorical Crossentropy   |
| Metric                  | Accuracy                   |
| Early Stopping          | Patience = 8               |
| Learning Rate Reduction | Factor = 0.5, Patience = 4 |

---

## Model Performance

### Results

| Metric            | Score |
| ----------------- | ----- |
| Baseline Accuracy | 92.1% |
| Tuned Accuracy    | 93.1% |

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

The model achieves strong classification performance across all activity classes, with minor confusion occurring between similar static activities such as Sitting and Standing.

---

## Explainable AI

To improve model transparency and interpretability, SHAP (SHapley Additive exPlanations) was integrated into the workflow.

The explainability module provides:

* Signed SHAP Analysis
* Magnitude-Based SHAP Analysis
* Feature Contribution Visualizations
* SHAP Heatmaps

These explanations help identify which sensor channels and time steps contribute most to a prediction, providing greater insight into model behavior.

---

## Real-Time Deployment

A Streamlit application was developed for live activity recognition using smartphone sensor data.

### Live Sensor Streaming with PhyPhox

The deployment pipeline uses:

* Android Smartphone Sensors
* PhyPhox Mobile Application
* PhyPhox Remote Server API
* Laptop Hotspot Connection
* CNN-LSTM Prediction Model
* Streamlit Dashboard

### Deployment Workflow

```text
Android Sensors
       ↓
     PhyPhox
       ↓
Remote Server API
       ↓
 Streamlit App
       ↓
 CNN-LSTM Model
       ↓
Activity Prediction
```

The smartphone is connected to the laptop through a local hotspot network. Sensor readings are streamed in real time using the PhyPhox Remote Server feature, allowing the trained CNN-LSTM model to perform live activity recognition and display predictions through the Streamlit dashboard.

---

## Streamlit Dashboard

The Streamlit interface provides:

* Real-Time Sensor Monitoring
* Live Activity Prediction
* Prediction Confidence Scores
* Interactive Visualization Dashboard

---

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Scikit-Learn
* SHAP
* Streamlit
* Matplotlib
* Keras Tuner
* PhyPhox

---

## Repository Structure

```text
human-activity-recognition-cnn-lstm/

├── app.py
├── cnn_lstm_model.h5
├── Gesture_Recognition.ipynb
├── requirements.txt
├── README.md
└── images/
```

---

## Future Improvements

* TensorFlow Lite Deployment
* Edge Deployment on Wearable Devices
* Gesture-Based Command Execution
* Industrial IoT Integration
* Multi-User Activity Recognition
* Real-Time Anomaly Detection

---

## Author

Navjot Singh
