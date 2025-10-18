# 🌲 Forest Cover Type Classification Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-yellow.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Clustering-orange.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-green.svg)
![Seaborn](https://img.shields.io/badge/Seaborn-EDA-blue.svg)



## 📌 Project Overview

This project focuses on supervised learning to predict the forest cover type from cartographic variables using different machine learning algorithms — Decision Tree, Random Forest, and XGBoost.
It aims to evaluate and compare model performances in terms of accuracy and generalization.

## 📂 Dataset

The dataset used is the [Forest Cover Type Dataset](https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset) Dataset
, which contains cartographic variables from the Roosevelt National Forest in northern Colorado.

- Features:-
  - 10 continuous features (e.g., Elevation, Aspect, Slope, etc.)
  - 44 binary features representing soil types and wilderness areas
  - Target: Cover_Type (7 forest cover types, represented as integers 1–7)

- Objective :-
  - Predict the forest cover type (e.g., Spruce/Fir, Lodgepole Pine, Aspen, etc.) based on environmental data.


## 📊 Results & Comparison

- **Decision Tree**
  - Training Accuracy: 0.9839  
  - Validation Accuracy: 0.9294  

- **Random Forest**
  - Training Accuracy: 0.9990  
  - Validation Accuracy: 0.9479  

- **XGBoost ✅**
  - Training Accuracy: **1.0000**  
  - Validation Accuracy: **0.9638**  
  - Test Accuracy: **0.9626**



## 🧩 Insights

- XGBoost achieved the highest test accuracy (96.26%), outperforming both Decision Tree and Random Forest.
- Random Forest performed well but was slightly behind XGBoost in validation accuracy.


## 🏁 Conclusion

The **XGBoost** model achieved the best overall performance with:

- **Training Accuracy:** 100%  
- **Validation Accuracy:** 96.38%  
- **Test Accuracy:** 96.26%

These results highlight the **effectiveness of gradient boosting** for structured classification tasks.  
Ensemble methods like **Random Forest** and **XGBoost** significantly outperform single decision trees in both **stability** and **accuracy**, making them ideal for real-world predictive modeling.


## ▶️ How to Run the Project

### 1. Prerequisites
- Python 3.8+  
- Git  
- Recommended: virtual environment to manage dependencies  

### 2. Clone the Repository
```bash
git clone https://github.com/OmarMoawad1112/forest-cover-type-classification.git
cd forest-cover-type-classification
```

### 3. Create and Activate a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate (Windows)
source venv/bin/activate (Mac/Linux)
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Project
```bash
python main.py
```
