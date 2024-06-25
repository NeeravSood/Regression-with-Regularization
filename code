import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV

def regression(data_train, data_test):
    # Prepare data
    X_train = data_train.drop(columns='price')
    y_train = data_train['price']
    X_test = data_test.drop(columns='price')
    
    # Define alphas and l1_ratios
    alphas = np.logspace(-4, -1, 4)
    l1_ratios = np.arange(0.6, 1, 0.1)
    
    # Initialize models
    ridge_model = RidgeCV(alphas=alphas, normalize=True, store_cv_values=True)
    lasso_model = LassoCV(alphas=alphas, normalize=True)
    elastic_net_model = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios, normalize=True)
    
    # Fit models
    ridge_model.fit(X_train, y_train)
    lasso_model.fit(X_train, y_train)
    elastic_net_model.fit(X_train, y_train)
    
    # Make predictions
    ridge_pred = np.round(ridge_model.predict(X_test), 2)
    lasso_pred = np.round(lasso_model.predict(X_test), 2)
    elastic_net_pred = np.round(elastic_net_model.predict(X_test), 2)
    
    # Extract coefficients and filter
    ridge_coefs = pd.DataFrame({
        'variable': X_train.columns,
        'coef': ridge_model.coef_
    })
    ridge_coefs = ridge_coefs[ridge_coefs['coef'].abs() > 0.001]
    
    lasso_coefs = pd.DataFrame({
        'variable': X_train.columns,
        'coef': lasso_model.coef_
    })
    lasso_coefs = lasso_coefs[lasso_coefs['coef'] != 0]
    
    elastic_net_coefs = pd.DataFrame({
        'variable': X_train.columns,
        'coef': elastic_net_model.coef_
    })
    elastic_net_coefs = elastic_net_coefs[elastic_net_coefs['coef'].abs() > 0.001]
    
    # Create result dictionary
    results = {
        'ridge': {
            'alpha': ridge_model.alpha_,
            'pred': ridge_pred,
            'coefficients': ridge_coefs
        },
        'lasso': {
            'alpha': lasso_model.alpha_,
            'pred': lasso_pred,
            'coefficients': lasso_coefs
        },
        'elastic_net': {
            'alpha': elastic_net_model.alpha_,
            'l1_ratio': elastic_net_model.l1_ratio_,
            'pred': elastic_net_pred,
            'coefficients': elastic_net_coefs
        }
    }
    
    return results
