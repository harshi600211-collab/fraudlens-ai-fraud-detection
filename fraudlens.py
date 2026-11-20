import numpy as np
import pandas as pd
 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
 
# 1. Generate a synthetic transaction dataset
np.random.seed(42)
rng = np.random.default_rng(42)
n = 5000
 
amount = np.round(
    rng.lognormal(mean=4.2, sigma=1.0, size=n), 2
)
hour = rng.integers(0, 24, n)
international = rng.binomial(1, 0.12, n)
velocity = rng.poisson(1.5, n)
new_device = rng.binomial(1, 0.18, n)
merchant_risk = rng.uniform(0, 1, n).round(3)
 
# Create a probability using sample risk factors
logit = (
    -5.0
    + 0.006 * np.minimum(amount, 1000)
    + 1.25 * international
    + 0.55 * velocity
    + 1.3 * new_device
    + 1.8 * merchant_risk
    + 0.9 * ((hour <= 5) | (hour >= 23))
)
probability = 1 / (1 + np.exp(-logit))
fraud = rng.binomial(1, probability, n)
 
data = pd.DataFrame({
    "amount": amount,
    "hour": hour,
    "international": international,
    "velocity": velocity,
    "new_device": new_device,
    "merchant_risk": merchant_risk,
    "fraud": fraud
})
 
print("Dataset Created Successfully!")
print(data.head())
 
# 2. Prepare features and target
X = data.drop("fraud", axis=1)
y = data["fraud"]
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y
)
 
# 3. Train the Random Forest classifier
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("\\nModel Training Completed!")

# 4. Evaluate the trained model

# 4. Evaluate the trained model
predictions = model.predict(X_test)
fraud_probabilities = model.predict_proba(X_test)[:, 1]
 
accuracy = accuracy_score(y_test, predictions)
auc = roc_auc_score(y_test, fraud_probabilities)
 
print("\\nModel Accuracy:", round(accuracy * 100, 2), "%")
print("ROC-AUC Score:", round(auc, 3))
print("\\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
 
print("\\nClassification Report:")
print(classification_report(y_test, predictions, digits=3))
 
# 5. Create example transactions for prediction
new_transactions = pd.DataFrame([
    [850, 2, 1, 5, 1, 0.92],
    [35, 14, 0, 0, 0, 0.12],
    [420, 23, 0, 3, 1, 0.75]
], columns=X.columns)
 
risk_scores = model.predict_proba(
    new_transactions
)[:, 1]
 
# 6. Convert probability into a risk category
def classify_risk(score):
    if score >= 0.70:
        return "HIGH RISK"
    elif score >= 0.30:
        return "MEDIUM RISK"
    return "LOW RISK"
 
results = new_transactions.copy()
results["Fraud Probability"] = risk_scores
results["Risk Level"] = [
    classify_risk(score) for score in risk_scores
]
 
print("\\nFraudLens Risk Analysis:")
print(results[
    ["amount", "Fraud Probability", "Risk Level"]
].round(3).to_string(index=False))
 
print("\\nFraudLens Analysis Completed!")
