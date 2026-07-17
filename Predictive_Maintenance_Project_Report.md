# Detailed Project Report
**Project Title:** AI-Powered Predictive Maintenance System for IoT Devices
**Domain:** Machine Learning, IoT, Industrial Analytics
**Frameworks Used:** Scikit-Learn, XGBoost, Streamlit, Plotly

---

## 1. Executive Summary
This project aims to develop a robust, industry-level Artificial Intelligence system capable of monitoring industrial machinery telemetry and predicting imminent equipment failures before they occur. Utilizing the NASA CMAPSS Turbofan Engine Degradation Dataset, the project demonstrates a complete end-to-end Machine Learning pipeline. The final deliverable includes both a dual-model predictive backend and a highly professional, PowerBI-style interactive web dashboard.

## 2. Problem Statement
In industrial sectors such as aviation, manufacturing, and energy, unexpected machinery breakdown leads to severe consequences:
- **Financial Loss:** Unplanned downtime costs millions in lost production.
- **Safety Hazards:** Critical failures (e.g., in aviation engines) pose risks to human life.
- **Maintenance Inefficiency:** Traditional "Preventative" maintenance replaces parts prematurely, wasting resources.

**Objective:** Transition from reactive/preventative maintenance to *Predictive Maintenance* by leveraging IoT sensor data to forecast the exact Remaining Useful Life (RUL) of equipment and issue warnings when failure is imminent.

## 3. Dataset Description
The project utilizes the **NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) Dataset**:
- **Data Structure:** Time-series telemetry data tracking engines from a healthy state until catastrophic failure.
- **Features:** 21 distinct sensor readings (e.g., temperatures, pressures, fan speeds) and 3 operational settings per engine per cycle (flight).
- **Target Variable formulation:** We derived two targets from the raw data:
  1. *Continuous RUL (Cycles)* - For regression tasks.
  2. *Binary Label (0 or 1)* - Where 1 indicates RUL <= 30 cycles (Failure Imminent).

## 4. Methodology
The development lifecycle was segmented into the following phases:
1. **Data Preprocessing:** Handled raw space-separated text files. Grouped data by Engine ID to calculate the maximum cycles and derive the RUL for every row.
2. **Feature Engineering:** Extracted relevant sensor columns and dropped non-predictive identifiers (like cycle index and ID) during the training phase.
3. **Train/Test Split:** Data was split 80/20 ensuring a stratified distribution of the binary failure labels.
4. **Model Evaluation:** Benchmarked three distinct algorithms for classification accuracy and regression mean absolute error (MAE).

## 5. Multi-Model Architecture
The predictive engine employs two parallel models to provide comprehensive insights:

### Classification Task (Anomaly Detection)
- **Logistic Regression:** Used as a baseline linear model. Achieved moderate accuracy.
- **Random Forest Classifier:** An ensemble learning method providing robust non-linear splits.
- **XGBoost Classifier:** Selected as the final model due to its gradient boosting framework, achieving **98.54% Accuracy** and **98.24% F1-score** in detecting the imminent 30-cycle failure window.

### Regression Task (RUL Forecasting)
- **XGBoost Regressor:** Deployed to predict the exact integer value of remaining cycles. This provides maintenance crews with an exact timeline for required interventions.

## 6. PowerBI-Style Web Dashboard
To ensure the machine learning insights are accessible to non-technical stakeholders (e.g., plant managers, maintenance crews), a highly professional web application was developed using **Streamlit** and **Plotly**.

**Dashboard Features:**
- **Custom UI/UX:** Styled using custom CSS to mimic enterprise dashboards (like PowerBI), utilizing modern fonts (`Inter`), custom KPI cards, and dynamic color coding (Green for Healthy, Red for Danger).
- **Interactive Controls:** Users can select specific Engine IDs and scrub through time (cycles) using a sidebar slider to simulate real-time IoT streaming.
- **Real-Time KPIs:** Prominently displays the Predicted RUL and Current Health Status.
- **Telemetry Visualization:** Plotly is used to render interactive line graphs of critical sensors (e.g., High-Pressure Compressor, Core Speed), allowing users to visually correlate sensor spikes with impending failure.

## 7. Results and Performance
- The **XGBoost Classifier** successfully distinguished between normal operations and imminent failure states with high test accuracy (**98.54%**) and high F1-score (**98.24%**), minimizing false alarms.
- The **XGBoost Regressor** achieved a Mean Absolute Error (MAE) of **45.79 cycles** on the combined holdout test set, providing highly reliable timelines for scheduled maintenance.
- **Evaluation Visualizations:** Three high-resolution plots were saved during pipeline training (`confusion_matrix.png`, `roc_curve.png`, and `rul_predictions.png` inside the `/outputs` directory) to validate model performance.
- The web dashboard successfully simulated real-time ingestion and inference with sub-second latency.

## 8. Conclusion
The AI-Powered Predictive Maintenance System successfully demonstrates how machine learning can transform raw IoT sensor data into actionable intelligence. By integrating state-of-the-art gradient boosting models with an enterprise-grade interactive dashboard, this project serves as a comprehensive blueprint for deploying predictive analytics in modern Industry 4.0 environments.
