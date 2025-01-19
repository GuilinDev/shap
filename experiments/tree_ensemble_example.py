import xgboost
import shap
import numpy as np

# Load data
X, y = shap.datasets.california()

# Train model
model = xgboost.train({"learning_rate": 0.01}, 
                     xgboost.DMatrix(X, label=y), 
                     100)

# Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Visualize
shap.summary_plot(shap_values, X) 
