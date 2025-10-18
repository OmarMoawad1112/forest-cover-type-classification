from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ================== LOAD DATA ==================
df = pd.read_csv("../data/covtype.csv")

columns = df.columns.tolist()
features = [x for x in columns if x != 'Cover_Type']
X = df[features]
y = df['Cover_Type']

binary_cols = [col for col in features if X[col].nunique() == 2]
non_binary_cols = [col for col in features if X[col].nunique() > 2]


# ================== SCALING FUNCTION ==================
def scale_continuous_features(X, non_binary_cols):
    """
    Scales continuous (non-binary) columns using StandardScaler and 
    merges them back with the remaining features.

    Args:
        X (pd.DataFrame): Original feature dataframe.
        non_binary_cols (list): List of continuous feature names to scale.

    Returns:
        pd.DataFrame: DataFrame with scaled continuous features and unchanged binary ones.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[non_binary_cols])
    X_scaled = pd.DataFrame(X_scaled, columns=non_binary_cols, index=X.index)

    # Merge scaled with unscaled features
    X_merged = pd.concat([X.drop(columns=non_binary_cols), X_scaled], axis=1)
    X_merged = X_merged[X.columns]  # keep original order

    return X_merged


# ================== SCALE FEATURES ==================
X = scale_continuous_features(X, non_binary_cols)


# ================== TRAIN / VALIDATION / TEST SPLIT ==================
# 1. Split into train + temp (which we'll later split again)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)  # 70% train, 30% temp

# 2. Split temp into validation + test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)  # 15% val, 15% test


# ================== LOAD TRAINED MODEL ==================
xgb_model_loaded = XGBClassifier()
xgb_model_loaded.load_model("../model/xgb_best_model.json")


# ================== EVALUATE MODEL ==================
y_pred_fixed = xgb_model_loaded.predict(X_test)
y_pred = y_pred_fixed + y_train.min()

print(f"The testing accuracy = {accuracy_score(y_pred, y_test)}")
