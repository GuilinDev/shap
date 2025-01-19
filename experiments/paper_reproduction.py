import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor
import time

def create_synthetic_data(n_samples=1000, n_features=10):
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] + 2 * X[:, 1] + np.random.randn(n_samples) * 0.1
    return X, y

def evaluate_shap_accuracy(true_importance, shap_values):
    """Evaluate the correlation between SHAP values and true feature importance"""
    mean_shap = np.abs(shap_values).mean(0)
    correlation = np.corrcoef(true_importance, mean_shap)[0,1]
    return correlation

if __name__ == "__main__":
    # Generate synthetic data
    X, y = create_synthetic_data()
    
    # Train random forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # True feature importance (in this example, [1, 2, 0, 0, ...])
    true_importance = np.zeros(X.shape[1])
    true_importance[0] = 1
    true_importance[1] = 2
    
    # Evaluate accuracy
    correlation = evaluate_shap_accuracy(true_importance, shap_values)
    print(f"Correlation with true importance: {correlation:.3f}") 
