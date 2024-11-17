import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


houses = pd.read_csv("houses.csv")

print(houses.head(6))

from sklearn.tree import DecisionTreeClassifier, export_text

X = houses[["dim_1", "dim_2"]]
y = houses["level"]

cl = DecisionTreeClassifier().fit(X, y)

print(export_text(cl))

new_data = [[120, 210], [180, 270], [160, 230]]
y_pred = cl.predict(new_data)
print(y_pred)