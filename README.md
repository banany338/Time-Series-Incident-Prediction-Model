# Incident Forecasting & Alerting System 🚀

This repository contains a production-oriented machine learning pipeline for predicting system incidents (e.g., threshold breaches or anomalies) using time-series metrics. The project was built with a focus on Site Reliability Engineering (SRE) principles, emphasizing early detection and the mitigation of "alert fatigue".

## 📌 Problem Formulation
The objective is to predict whether an incident will occur within a future horizon of $H=5$ time steps, based on a sliding lookback window of $W=20$ previous time steps. 

Given the highly imbalanced nature of system anomalies, simple accuracy is an inadequate metric. This project optimizes for **business value**, using Cost-Sensitive Learning to balance Precision (reducing false alarms) and Recall (catching critical failures).

> **Note on Data:** This project currently utilizes a highly noisy, synthetic dataset to simulate CPU and Memory metrics with injected degradation patterns. However, the feature engineering and modeling pipeline is entirely data-agnostic and can be seamlessly adapted to ingest real-world telemetry data from observability stacks like Prometheus or Datadog.

## ⚙️ Architecture & Design Decisions

### 1. Data Generation & Feature Engineering
* **Multivariate Time-Series:** Simulates a realistic environment with correlated metrics and injects realistic anomaly patterns (sudden spikes and gradual memory leaks).
* **Rolling Statistics & Anomaly Scores:** Instead of relying purely on raw values, the model extracts deep features for each window:
    * Trend analysis using linear regression slopes (`np.polyfit`).
    * Rolling percentiles (25th, 75th) to capture spread.
    * Deviation from the local mean (a localized Anomaly Score).
    * Cross-metric interactions (CPU/Memory ratios).

### 2. Model Selection & Temporal Validation
* **XGBoost Classifier:** Chosen for its robustness to non-linear relationships and built-in support for extreme class imbalances via `scale_pos_weight` and `max_delta_step`.
* **TimeSeriesSplit:** Standard cross-validation causes data leakage in time-series tasks. Hyperparameter tuning (`GridSearchCV`) was performed using chronological splits to ensure the model only learns from the past to predict the future.
* **Probability Calibration:** Tree-based models often yield uncalibrated probabilities. `CalibratedClassifierCV` (using Platt Scaling/Sigmoid) was applied to ensure that the outputted anomaly scores reflect true probabilities.

### 3. Cost-Sensitive Thresholding (The Business Logic)
To combat alert fatigue without compromising system safety, the decision threshold was not set to a default $0.5$, nor was it optimized purely for F1-score. Instead, a custom business cost function was introduced:
* **Cost of False Positive ($FP$):** 1 (e.g., an engineer investigates a false alarm).
* **Cost of False Negative ($FN$):** 5 (e.g., an outage occurs, breaching SLA).

The model dynamically sweeps the Precision-Recall curve on the training set to find the exact probability threshold that minimizes the total operational cost.

## 📊 Results & Performance

Based on the 1:5 business cost ratio, the optimal probability threshold was calculated at **0.3090**. Evaluated on the held-out test set (30% of the timeline), the model achieved the following:

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0 (Normal)** | 0.91 | 0.79 | 0.85 | 2560 |
| **1 (Incident)** | **0.30** | **0.54** | 0.39 | 433 |

**Business Interpretation:** The model successfully catches over half (54%) of all incidents in a highly noisy environment. At a 30% precision rate, roughly 1 in 3 alerts sent to operators is a genuine incident. In a real-world scenario, this raw output would be further filtered by alert debouncing (requiring consecutive positive flags) to drive the final operational precision even higher.

## 🚀 Future Work & Adaptations
Deploying this pipeline into a live SRE environment would involve the following enhancements:

* **Alert Debouncing:** Smoothing raw predictions to prevent alert flapping (e.g., firing a PagerDuty alert only if the model predicts an incident for 3 consecutive time steps).
* **Sequence-Aware Architectures:** Transitioning from tree-based windowing to explicit temporal models like LSTMs (Long Short-Term Memory) or TCNs (Temporal Convolutional Networks) to better capture complex sequential degradation.
* **Survival Analysis:** Shifting the target variable from a binary classification ("incident in next $H$ steps") to a regression task ("time-to-incident") to provide richer early warnings.

## 🛠️ How to Run

```bash
# Clone the repository
git clone github.com/banany338/Time-Series-Incident-Prediction-Model
cd [your path]/Time-Series-Incident-Prediction-Model

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python main.py
