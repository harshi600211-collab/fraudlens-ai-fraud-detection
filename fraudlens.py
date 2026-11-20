model.fit(X_train, y_train)
print("\nModel Training Completed!")

# 4. Evaluate the trained model

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nFRAUD DETECTION MODEL RESULTS")
print("----------------------------")
print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("ROC-AUC Score:", round(roc_auc_score(y_test, y_prob), 4))


# 5. Test a new financial transaction

new_transaction = pd.DataFrame([{
    "amount": 9500.00,
    "hour": 2,
    "international": 1,
    "velocity": 8,
    "new_device": 1,
    "merchant_risk": 0.95
}])

fraud_probability = model.predict_proba(new_transaction)[0][1]
risk_score = round(fraud_probability * 100)

if risk_score >= 70:
    risk_category = "HIGH RISK"
    action = "Hold transaction for manual review"
elif risk_score >= 40:
    risk_category = "MEDIUM RISK"
    action = "Request additional verification"
else:
    risk_category = "LOW RISK"
    action = "Allow normal processing"

print("\nNEW TRANSACTION ANALYSIS")
print("------------------------")
print("Fraud Probability:", round(fraud_probability, 4))
print("Risk Score:", str(risk_score) + "/100")
print("Risk Category:", risk_category)
print("Recommended Action:", action)
