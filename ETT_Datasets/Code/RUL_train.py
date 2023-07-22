import pandas as pd
from datetime import datetime, timedelta
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from scipy import interpolate

train = pd.read_csv("ETTm2_train.csv", usecols=["date","OT"])
train['date'] = pd.to_datetime(train['date'])
train['date'] = train["date"].to_numpy()

print(train['date'])

x = train.iloc[0:2976,0].values  
y = train.iloc[0:2976,1].values

# # fit = np.polyfit(x, y, 3)
# # line = np.poly1d(fit)

# # line()
# # time_list = []
# # for i in range (2520):
# #     np.datetime64(x[-1]) + np.timedelta64(15, 'm')
# #     time_list.append()

# # new_points
   
model = PolynomialFeatures(degree=16, include_bias = False)
X_poly = model.fit_transform(x.reshape(-1, 1))
poly_reg = LinearRegression()
p = poly_reg.fit(X_poly, y)
y_predict = p.predict(X_poly)

max_threshold = 70
min_threshold = 0

#plt.rcParams["figure.figsize"] = [7.50, ]
plt.scatter(x, y, color='red', label='Training data')
plt.plot(x, p, color='green', label='predicted')
plt.axhline( max_threshold, color = 'r', linestyle = '-', label='max threshold')
plt.axhline( min_threshold, color = 'r', linestyle = '-', label='min threshold')
plt.title("OT change")
plt.xlabel('date')
plt.ylabel('OT')
plt.legend()
plt.show()      






# for i in range(len(train)):




#     current_date = train['date'][i]
#     current_ot = train["OT"][i]

#     start_date = train["date"].min()
#     train['DaysSinceStart'] = (train["date"][i] - start_date).dt.days
#     X = train[["DaysSinceStart", "OT"]]
#     Y = train["DaysSinceStart"]

# for index, row in train.iterrows():
#     current_date = row['date']  # Get the current date for the row
#     current_oil_temp = row['OT']  # Get the current oil temperature for the row
    
#     # Perform the remaining steps for feature engineering and model training using the current row data
#     start_date = train['date'].min()
#     train['DaysSinceStart'] = (train['date'] - start_date).dt.days
#     x = train['DaysSinceStart']
#     y = train['OT']


# days_since_start = (current_date - start_date).days
# predicted_rul = model.predict([[days_since_start, current_oil_temp]])
# print(predicted_rul)

