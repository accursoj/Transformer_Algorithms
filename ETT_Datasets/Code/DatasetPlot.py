import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
columns = ["date", "OT"]
transformer = pd.read_csv("ETTm1_month1.csv", usecols=columns)
plt.plot(transformer.date, transformer.OT)
plt.show()

transformer = pd.read_csv("ETTm2_month1.csv", usecols=columns)
plt.plot(transformer.date, transformer.OT)
plt.show()