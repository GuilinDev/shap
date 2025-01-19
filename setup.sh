#!/bin/bash

# Check if Xcode Command Line Tools are installed
if ! command -v xcode-select &> /dev/null; then
    echo "Installing Xcode Command Line Tools..."
    xcode-select --install
    # Wait for user to complete installation in the popup window
    read -p "Press enter after the installation has completed"
fi

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install OpenMP
echo "Installing OpenMP runtime..."
brew install libomp

# Create and activate virtual environment if not already done
# python -m venv .venv
# source .venv/bin/activate

# Install basic dependencies
echo "Installing basic dependencies..."
pip install numpy scipy scikit-learn pandas tqdm packaging slicer numba cloudpickle typing-extensions

# Install visualization dependencies
echo "Installing visualization dependencies..."
pip install matplotlib

# Install ML frameworks
echo "Installing ML frameworks..."
pip install xgboost lightgbm

# Try installing tensorflow (optional)
echo "Attempting to install tensorflow (optional)..."
pip install tensorflow || echo "Tensorflow installation failed, but this is optional for basic usage"

# Install SHAP from source
echo "Installing SHAP from source..."
pip install --editable '.[plots]' 
