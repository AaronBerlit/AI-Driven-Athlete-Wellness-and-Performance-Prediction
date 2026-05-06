# 🏅 AI-Driven Athlete Wellness and Performance Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Mobile](https://img.shields.io/badge/Mobile-TFLite%20Optimized-red.svg)

An advanced deep learning framework designed to monitor, analyze, and predict athlete performance, readiness, and nutritional needs. This project leverages neural networks and Natural Language Processing (NLP) to provide actionable insights for professional athletes and fitness enthusiasts.

---

## 🚀 Project Overview

This repository contains a multi-model pipeline that transforms raw athletic data into predictive intelligence. The system is divided into three core pillars:

1.  **Performance Model**: A regression neural network that calculates a performance score (0–100) based on training intensity, physiological markers (BPM), and historical activity.
2.  **Readiness (Wellness) Model**: A classification system that determines an athlete's mental and physical readiness state by analyzing sleep patterns, stress levels, and lifestyle factors.
3.  **Nutrition Model**: An NLP-powered regression model that uses `SentenceTransformers` to predict detailed nutrient profiles from simple food descriptions.

---

## ✨ Key Features

*   **Deep Learning Architectures**: Custom-built Keras/TensorFlow models optimized for high accuracy.
*   **NLP Embeddings**: Uses `all-MiniLM-L6-v2` for semantic understanding of nutritional data.
*   **Edge-Ready (Mobile Optimization)**: Includes full support for **TensorFlow Lite (TFLite)** with Float16 and Int8 quantization for deployment on mobile devices.
*   **Robust Data Pipelines**: Automated preprocessing using `Scikit-Learn` pipelines for scaling, one-hot encoding, and handling categorical variables.
*   **Visual Analytics**: Built-in tools for training visualization, confusion matrices, and residual analysis.

---

## 📂 Project Structure

```text
├── dataset/
│   ├── dataset_performancemodel.csv  # Performance tracking data
│   ├── nutrition_cleaned.csv        # Food nutrient database
│   └── readiness_datasets.csv       # Wellness and lifestyle logs
├── Deeplearining(code).py           # Core training and conversion script
├── performance_model.tflite         # Optimized performance model
├── readiness_model.tflite           # Optimized readiness model
├── nutrition_model.tflite           # Optimized nutrition model
└── target_scaler.pkl                # Saved scaler for nutrition data
```

---

## 🛠️ Installation & Setup

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/AaronBerlit/AI-Driven-Athlete-Wellness-and-Performance-Prediction.git
    cd AI-Driven-Athlete-Wellness-and-Performance-Prediction
    ```

2.  **Install Dependencies**:
    ```bash
    pip install tensorflow scikit-learn pandas numpy matplotlib seaborn sentence-transformers joblib
    ```

3.  **Run the Training Pipeline**:
    ```bash
    python "Deeplearining(code).py"
    ```

---

## 📊 Models & Methodology

### 1. Performance Prediction
*   **Input**: Age, Weight, Height, Avg/Max BPM, Calories Burned, Session Hours, etc.
*   **Target**: `performance_score` (Synthetic weighted metric).
*   **Optimization**: Adam optimizer with MSE loss.

### 2. Readiness Assessment
*   **Input**: Sleep hours, Stress Level, BMI, Blood Pressure, Mood, Workload.
*   **Target**: Categorical readiness states.
*   **Processing**: Handles complex string formats (e.g., "120/80" BP) and transforms them into numerical features.

### 3. AI Nutritionist
*   **Input**: Raw food names (e.g., "Mushrooms, raw, enoki").
*   **Target**: Multi-output nutrient values (Protein, Fat, Carbs, etc.).
*   **NLP**: Converts text to 384-dimensional vectors for deep learning analysis.

---

## 📱 Mobile Deployment

All models are exported in `.tflite` format. We provide:
*   **Standard TFLite**: Balanced performance.
*   **Quantized TFLite**: Significantly reduced model size (up to 4x smaller) with minimal accuracy loss, perfect for real-time mobile apps.

---

## 📈 Evaluation

The project includes comprehensive evaluation metrics:
*   **Regression**: MSE, MAE, and R² Scores.
*   **Classification**: Precision, Recall, F1-Score, and Confusion Matrices.
*   **Efficiency**: Comparison graphs between original and quantized TFLite models.

---

## 🤝 Contributing

Contributions are welcome! If you have ideas for new features or improvements, feel free to fork the repo and create a pull request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---
*Created with ❤️ for the Athlete Community.*
