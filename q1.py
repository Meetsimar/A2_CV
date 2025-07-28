# Import necessary libraries
import pandas as pd
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from math import sqrt

# Load the dataset
df = pd.read_csv("house_price.csv")
print(df.head())  # Print first few rows to understand the data structure

# Define features (bedrooms and size) and target (price)
X = df[["bedroom", "size"]]
y = df["price"]

### ----- Linear Regression -----
# Train a basic Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X, y)
y_pred_lr = lr_model.predict(X)

# Display results
print("\n----- Linear Regression -----")
print("Coefficients (bedroom, size):", lr_model.coef_)
print("Intercept:", lr_model.intercept_)

# Calculate error metrics
mae_lr = mean_absolute_error(y, y_pred_lr)
mse_lr = mean_squared_error(y, y_pred_lr)
rmse_lr = sqrt(mse_lr)
mape_lr = mean_absolute_percentage_error(y, y_pred_lr)

print(f"MAE: {mae_lr:.2f}")
print(f"MSE: {mse_lr:.2f}")
print(f"RMSE: {rmse_lr:.2f}")
print(f"MAPE: {mape_lr:.4f}")

### ----- SGD Regressor (with Scaling) -----
# Standardize the feature values before using SGD
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train SGD Regressor (a linear model trained with stochastic gradient descent)
sgd_model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
sgd_model.fit(X_scaled, y)
y_pred_sgd = sgd_model.predict(X_scaled)

# Show SGD results
print("\n----- SGD Regressor (with scaled features) -----")
print("Coefficients (bedroom, size):", sgd_model.coef_)
print("Intercept:", sgd_model.intercept_)

# Calculate error metrics
mae_sgd = mean_absolute_error(y, y_pred_sgd)
mse_sgd = mean_squared_error(y, y_pred_sgd)
rmse_sgd = sqrt(mse_sgd)
mape_sgd = mean_absolute_percentage_error(y, y_pred_sgd)

print(f"MAE: {mae_sgd:.2f}")
print(f"MSE: {mse_sgd:.2f}")
print(f"RMSE: {rmse_sgd:.2f}")
print(f"MAPE: {mape_sgd:.4f}")

### ----- Explanation -----
# Print explanation about error metrics
print("""
Trade-offs between MAE, MSE, and RMSE:

- MAE (Mean Absolute Error): 
  Measures average error in same units as target. Robust to outliers. Easy to interpret.

- MSE (Mean Squared Error): 
  Squares the errors, penalizes large mistakes more. Sensitive to outliers.

- RMSE (Root Mean Squared Error): 
  Same units as target. Balances interpretability and sensitivity to big errors.

- MAPE (Mean Absolute Percentage Error): 
  Measures error as a percentage. Useful for comparing models across datasets, 
  but unstable if actual values are near zero.
""")
