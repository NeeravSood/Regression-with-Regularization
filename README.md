# Regression Models with Regularization

This repository contains Python code that implements linear regression models with three types of regularization: Ridge, Lasso, and Elastic Net. The models are built using the `sklearn` library and include cross-validation to select the best hyperparameters (`alpha` and `l1_ratio`).

### Assignment Overview

In this assignment, we developed a function `regression()` that accepts training and test data in the form of pandas DataFrames. The function performs the following tasks for each regression model:

- **Ridge Regression:**
  - Uses `RidgeCV` from `sklearn` with built-in cross-validation to select the optimal regularization parameter (`alpha`). The predictions and selected coefficients are returned.

- **Lasso Regression:**
  - Utilizes `LassoCV` to perform L1 regularization. Similar to Ridge, it selects the best `alpha`, makes predictions, and filters out coefficients that are effectively zero.

- **Elastic Net Regression:**
  - Implements `ElasticNetCV`, which combines L1 and L2 regularization. It selects both `alpha` and `l1_ratio` (mixing parameter between L1 and L2 penalties), makes predictions, and filters coefficients.

### File Structure

- `regression.py`: Python script containing the `regression()` function implementation.
- `README.md`: This file, providing an overview of the project, its purpose, and usage instructions.

### Usage

To use the `regression()` function:

1. Ensure you have Python 3.8+ installed along with the necessary libraries (`pandas`, `numpy`, and `scikit-learn`).
   
2. Include your training and test data as pandas DataFrames in the function call.
   
   ```python
   from regression import regression
   import pandas as pd
   
   # Example usage
   data_train = pd.read_csv('data_train.csv')  # Replace with your training data
   data_test = pd.read_csv('data_test.csv')    # Replace with your test data
   
   results = regression(data_train, data_test)
   ```
   
3. The function returns a dictionary `results` containing predictions and coefficients for Ridge, Lasso, and Elastic Net regression models.

### Dependencies

- Python 3.8+
- pandas 1.0+
- numpy 1.18+
- scikit-learn 0.23+
