# CUSTOMER CHURN PREDICTION

# Import libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------------
# STEP 1 : Load Dataset
# -----------------------------------

# Replace with your dataset file name
data = pd.read_csv("churn.csv")

# Display first 5 rows
print("First 5 Rows")
print(data.head())

# Dataset information
print("\nDataset Info")
print(data.info())

# Check missing values
print("\nMissing Values")
print(data.isnull().sum())

# -----------------------------------
# STEP 2 : Data Preprocessing
# -----------------------------------

# Convert categorical columns into numeric
label_encoder = LabelEncoder()

for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = label_encoder.fit_transform(data[column])

# Features and Target
X = data.drop("Churn", axis=1)
y = data["Churn"]

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------
# STEP 3 : Logistic Regression
# -----------------------------------

print("\n===== Logistic Regression =====")

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, lr_pred))

print("\nClassification Report")
print(classification_report(y_test, lr_pred))

# -----------------------------------
# STEP 4 : Random Forest
# -----------------------------------

print("\n===== Random Forest =====")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, rf_pred))

print("\nClassification Report")
print(classification_report(y_test, rf_pred))

# -----------------------------------
# STEP 5 : Gradient Boosting
# -----------------------------------

print("\n===== Gradient Boosting =====")

gb_model = GradientBoostingClassifier()

gb_model.fit(X_train, y_train)

gb_pred = gb_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, gb_pred))

print("\nClassification Report")
print(classification_report(y_test, gb_pred))

# -----------------------------------
# STEP 6 : Confusion Matrix
# -----------------------------------

cm = confusion_matrix(y_test, rf_pred)

sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues')

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# -----------------------------------
# STEP 7 : Compare Accuracy
# -----------------------------------

print("\n===== Model Comparison =====")

print("Logistic Regression Accuracy:",
      accuracy_score(y_test, lr_pred))

print("Random Forest Accuracy:",
      accuracy_score(y_test, rf_pred))

print("Gradient Boosting Accuracy:",
      accuracy_score(y_test, gb_pred))