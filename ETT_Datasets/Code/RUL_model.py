import pandas as pd
import datetime as dt
import numpy as np
import time 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from scipy import interpolate

#loading dataset(1st month is 0 - 2976 rows)
train = pd.read_csv("ETTm2_train.csv", usecols=["date","OT"], nrows= 2976)
train['date'] = pd.to_datetime(train['date'])
train['date'] = train['date'].apply(lambda x: dt.datetime.strptime(str(x),'%Y-%m-%d %H:%M:%S').timestamp())

x = train['date'].values
y = train['OT'].values

#Splitting data( test and train)
train.fillna(method ='ffill', inplace = True)
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, stratify=None, test_size = 0.40)

#Polynomial Regression Line
model = PolynomialFeatures(degree=4, include_bias = False)
x_poly = model.fit_transform(x_train.reshape(-1, 1))
x_poly_test = model.transform(x_test.reshape(-1, 1))
model.fit(x_poly, y_train)
lin_reg = LinearRegression()
lin_reg.fit(x_poly, y_train)
y_predict = lin_reg.predict(x_poly)
y_predict_test = lin_reg.predict(x_poly_test)

mean_abs = mean_absolute_error(y_train, y_predict)
print("Mean Absolute Error:", mean_abs)
# mean_abs = mean_absolute_error(y_test, y_predict_test)
# print("Mean Absolute Error:", mean_abs)

# #Extrapolating Regression line
fit = np.polyfit(x_train, y_predict, 4)
line = np.poly1d(fit)
new_time_points = np.arange(x_train[len(x_train)-1], x_train[len(x_train)-1] + 5184000, 900, int)
extrapolated_points = line(new_time_points)

#threshold values
max_threshold = 65
min_threshold = 0

#Stopping extrapolation once it hits threshold
extrap_filt = []
new_x = []
for i in range(len(extrapolated_points)):
    if extrapolated_points[i] < max_threshold and extrapolated_points[i] > min_threshold:
        extrap_filt.append(extrapolated_points[i])
        new_x.append(new_time_points[i])

#RUL Calculation
RUL = new_x[-1] - x_train[-1]
RUL = round(RUL/86400)
print(f"The RUL is {RUL} days")

#visualing
plt.plot(x_train, y_train, color='red', label='Training data')
plt.plot(x_train, y_predict, color='green', label='predicted')
plt.plot(new_x, extrap_filt, color='purple', label='Extrapolation')
plt.axhline( max_threshold, color = 'r', linestyle = '-', label='max threshold')
plt.axhline( min_threshold, color = 'r', linestyle = '-', label='min threshold')
plt.axvline(x = x_train[-1], color = 'blue', linestyle= '-',label='Last Value of x_train')
plt.title("OT change")
plt.xlabel('time')
plt.ylabel('OT')
plt.legend()
plt.show()
