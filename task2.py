import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. GENERATE SYNTHETIC DATASET (Simulating real credit card transaction patterns)
print("--- Step 1: Generating Dataset ---")
np.random.seed(42)
n_samples = 5000

# Features: Transaction Amount, Distance from Home, Time of Day
amount = np.random.exponential(scale=50, size=n_samples)
distance = np.random.lognormal(mean=1.5, sigma=0.8, size=n_samples)
time_of_day = np.random.uniform(0, 24, size=n_samples)

# Simple fraud logic simulation (Higher amounts + massive distances = high fraud probability)
fraud_probability = 1 / (1 + np.exp(-(0.01 * amount + 0.05 * distance - 3)))
is_fraud = np.random.binomial(1, fraud_probability)

# Assemble into a clean DataFrame
df = pd.DataFrame({
    'Amount': amount,
    'Distance_From_Home': distance,
    'Time_Of_Day': time_of_day,
    'Class': is_fraud
})

print(f"Dataset generated with {n_samples} transactions.")
print(f"Fraudulent cases identified: {df['Class'].sum()} ({~df['Class'].mean()*100:.2f}%)\n")

# 2. DATA PREPROCESSING
print("--- Step 2: Preprocessing Data ---")
X = df.drop(columns=['Class'])
y = df['Class']

# Split into training and testing sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical features for better model stability
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data splitting and scaling complete.\n")

# 3. MODEL TRAINING
print("--- Step 3: Training Random Forest Classifier ---")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)
print("Model training completed successfully.\n")

# 4. MODEL EVALUATION
print("--- Step 4: Performance Evaluation ---")
y_pred = model.predict(X_test_scaled)

# Output evaluation metrics
print("Accuracy Score:")
print(f"{accuracy_score(y_test, y_pred):.4f}\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
