import xgboost
import shap
import numpy as np
import time
import pandas as pd

def compare_explainers(X, y, model):
    results = {}
    
    # Tree SHAP
    start = time.time()
    tree_explainer = shap.TreeExplainer(model)
    tree_values = tree_explainer.shap_values(X)
    results['TreeExplainer'] = time.time() - start
    
    # Kernel SHAP
    start = time.time()
    def model_predict(x):
        # Ensure x is a DataFrame and has correct column names
        if isinstance(x, np.ndarray):
            x = pd.DataFrame(x, columns=X.columns)
        dmatrix = xgboost.DMatrix(x)
        return model.predict(dmatrix)
    
    # Use a subset of the original data as background data
    background_data = X.sample(n=100, random_state=42)
    kernel_explainer = shap.KernelExplainer(model_predict, background_data)
    kernel_values = kernel_explainer.shap_values(X.iloc[:100])  # Use iloc to maintain DataFrame format
    results['KernelExplainer'] = time.time() - start
    
    return results, tree_values, kernel_values

if __name__ == "__main__":
    # Load data
    X, y = shap.datasets.california()
    
    # Train model
    model = xgboost.train({"learning_rate": 0.01}, 
                         xgboost.DMatrix(X, label=y), 
                         100)
    
    # Compare different explainers
    timing_results, tree_shap, kernel_shap = compare_explainers(X, y, model)
    
    # Print results
    print("Timing results:")
    for explainer, time_taken in timing_results.items():
        print(f"{explainer}: {time_taken:.2f} seconds")
    
    # Optional: Add result visualization
    print("\nTree SHAP vs Kernel SHAP comparison for first instance:")
    print(f"Tree SHAP values: {tree_shap[0][:5]}")  # Show first 5 features
    print(f"Kernel SHAP values: {kernel_shap[:5]}")  # Show first 5 features 
