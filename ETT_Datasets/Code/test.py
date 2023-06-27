import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
  
'''
# Extract the desired columns from the DataFrame
column1_array = dataset['date'].values
column2_array = dataset['OT'].values

# Reshape the columns using array.reshape(-1, 1)
reshaped_array = np.column_stack((column1_array.reshape(-1, 1), column2_array.reshape(-1, 1)))
reshaped_dataset = pd.DataFrame(reshaped_array, columns=['DATE','OT'])
print (reshaped_dataset)

#Convert date to datetime
from datetime import datetime, timedelta
reshaped_dataset['DATE'] = pd.to_datetime(reshaped_dataset['DATE'], format="%m/%d/%Y %H:%M")
print(reshaped_dataset)
'''
dataset = pd.read_csv('ETTm1_month1.csv')
dataset
 
X = dataset.iloc[:,1:2].values  
y = dataset.iloc[:,7].values
 
# fitting the linear regression model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)
 
# visualising the linear regression model
plt.scatter(X,y, color='red')
plt.plot(X, lin_reg.predict(X),color='blue')
plt.title("OT change based on HUFL")
plt.xlabel('High Useful Load')
plt.ylabel('OT')
plt.show()

#define predictor and response variables
X, y = dataset[["hours", "prep_exams"]], dataset.score

#fit regression model
lin_reg.fit(X, y)

#calculate R-squared of regression model
r_squared = lin_reg.score(X, y)

#view R-squared value
print(r_squared)

 
# polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
  
X_poly     # prints X_poly
 
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)
 
 
# visualising polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)
  
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1) 
plt.scatter(X,y, color='red') 
  
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)),color='blue') 
  
plt.title("OT change based on HUFL")
plt.xlabel('High Useful Load')
plt.ylabel('OT')
plt.show()