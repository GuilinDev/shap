import shap
import xgboost
import numpy as np

# Use California Housing dataset instead of Boston
X, y = shap.datasets.california()
print("Data shape:", X.shape)
print("Data loaded successfully!")

# Simple model training test
model = xgboost.train({"learning_rate": 0.01}, 
                     xgboost.DMatrix(X, label=y), 
                     num_boost_round=10)
print("Model training successful!")

# Test SHAP values calculation
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X[:100])  # Use first 100 samples for testing
print("SHAP values calculation successful!")
