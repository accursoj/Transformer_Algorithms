import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
  
dataset = pd.read_csv('ETTm1_month1.csv')
dataset
 
X = dataset.iloc[:,1:2].values  
y = dataset.iloc[:,7].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression model
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred = lin_model.predict(X_train)

# Calculate the R-squared value using the training set
r_squared = r2_score(y_train, y_pred)
print("R-squared value:", r_squared)

# Visualizing Linear Regression
X_grid = np.arange(min(X_train), max(X_train), 0.1).reshape(-1, 1)
plt.scatter(X_train, y_train, color='red', label='Training data')
plt.scatter(X_test, y_test, color='blue', label='Testing data')
plt.plot(X_grid, lin_model.predict(X_grid), color='green', label='Linear regression')
plt.title("OT change based on HUFL")
plt.xlabel('High Useful Load')
plt.ylabel('OT')
plt.legend()
plt.show()


# Polynomial regression model
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X_train)
lin_model2 = LinearRegression()
lin_model2.fit(X_poly, y_train)
y_pred = lin_model2.predict(X_poly)

# Calculate the R-squared value using the test set
r_squared = r2_score(y_train, y_pred)
print("R-squared value:", r_squared)

# Visualizing polynomial regression
X_grid = np.arange(min(X_train), max(X_train), 0.1).reshape(-1, 1)
X_poly_grid = poly_reg.transform(X_grid)

plt.scatter(X_train, y_train, color='red', label='Training data')
plt.scatter(X_test, y_test, color='blue', label='Testing data')
plt.plot(X_grid, lin_model2.predict(X_poly_grid), color='green', label='Polynomial regression')
plt.title("OT change based on HUFL")
plt.xlabel('High Useful Load')
plt.ylabel('OT')
plt.legend()
plt.show()

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_train)

# Calculate the R-squared value using the training set
r_squared = r2_score(y_train, y_pred)
print("R-squared value:", r_squared)

# Visualizing Random Forest regression
X_grid = np.arange(min(X_train), max(X_train), 0.1).reshape(-1, 1)

plt.scatter(X_train, y_train, color='red', label='Training data')
plt.scatter(X_test, y_test, color='blue', label='Testing data')
plt.plot(X_grid, rf.predict(X_grid), color='green', label='Random Forest regression')
plt.title("OT change based on HUFL")
plt.xlabel('High Useful Load')
plt.ylabel('OT')
plt.legend()
plt.show()
