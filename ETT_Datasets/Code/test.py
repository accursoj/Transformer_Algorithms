import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
  
dataset = pd.read_csv('ETTh1_month1.csv')
dataset
 
X = dataset.iloc[:,1:2].values  
y = dataset.iloc[:,7].values
 
# Linear Regression
lin_model = LinearRegression()
lin_model.fit(X,y)
y_pred = lin_model.predict(X)
# Calculate the R-squared value
r_squared = r2_score(y, y_pred)

print("R-squared value:", r_squared)
# visualising the linear regression model
plt.scatter(X,y, color='red')
plt.plot(X, lin_model.predict(X),color='blue')
plt.title("OT change based on HUFL")
plt.xlabel('High Useful Load')
plt.ylabel('OT')
plt.show()


 
# polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X) 
X_poly     # prints X_poly 
lin_model2 = LinearRegression()
lin_model2.fit(X_poly,y)
y_pred = lin_model2.predict(X_poly)

# Calculate the R-squared value
r_squared = r2_score(y, y_pred)
print("R-squared value:", r_squared)

# visualising polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_model2 = LinearRegression()
lin_model2.fit(X_poly,y)
  
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1) 
plt.scatter(X,y, color='red') 
  
plt.plot(X_grid, lin_model2.predict(poly_reg.fit_transform(X_grid)),color='blue') 
  
plt.title("OT change based on HUFL")
plt.xlabel('High Useful Load')
plt.ylabel('OT')
plt.show()
