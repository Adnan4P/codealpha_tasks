# Multi-Model AI Pipeline Repository

This repository is a collection of my two primary AI-based projects: **Credit Scoring Pipeline** and **Disease Prediction System**. Both projects aim to generate intelligent predictions and insights from real-world data.

## 1. Credit Scoring Pipeline
This project is a robust machine learning pipeline designed to analyze financial data for loan risk scoring[cite: 1].

### Key Features:
* **Data Pipeline:** Includes automated data loading, cleaning, and preprocessing[cite: 1].
* **Feature Engineering:** Utilizes custom features to improve model accuracy[cite: 1].
* **Model Selection:** Employs structured evaluation metrics to select the optimal model during training[cite: 1].

### Project Structure:
* `main.py`: The core entry point of the pipeline that manages everything from data acquisition to evaluation[cite: 1].
* `src/`: Contains modular code including scripts for data loading, preprocessing, feature engineering, model building, and evaluation[cite: 1].

---

## 2. Disease Prediction System
This is a Flask-based web application that provides predictions and feature interpretability for heart disease, diabetes, and breast cancer[cite: 2].

### Key Features:
* **API-First Design:** Enables seamless predictions through the `/api/predict` endpoint[cite: 2].
* **Interpretability:** Beyond just predictions, the system uses a `contributions` list to explain which features (e.g., age, BP) are significantly impacting the model's output[cite: 2].
* **Dynamic Loading:** Models and scalers are loaded during system startup to minimize latency[cite: 2].

### How to Run:
1. Ensure your `.joblib` files and `metrics_summary.json` are present in the `models/` directory[cite: 2].
2. Run the `python app.py` command[cite: 2].
3. To test the API endpoints, send a `POST` request to `http://localhost:5000/api/predict`[cite: 2].

---

## Technical Stack
* **Language:** Python
* **Libraries:** Scikit-Learn, Pandas, NumPy, Flask, Joblib
* **Architecture:** Modular pipeline design, RESTful API integration.

---

## Author
**Muhammad Adnan**
*Student, Bachelor of Science in Artificial Intelligence*