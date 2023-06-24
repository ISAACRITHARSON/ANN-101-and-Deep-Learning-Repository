import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('Training Data for Linear Regression.csv')
test = pd.read_csv('Testing Data for Linear Regression.csv')
# training variables
x_train = train[['Max Temp']]
y_train = train['Solar Radiation']
# testing variables

x_test = test[['Max Temp']]
y_test = test['Solar Radiation']
from sklearn.linear_model import LinearRegression as lr
model = lr()
model.fit(x_train,y_train) y_pred = model.predict(x_test)
from sklearn.metrics import mean_absolute_error as mas
import math
math.sqrt(mas(y_test,y_pred))
plt.scatter(x_test,y_test,marker='^',label='Actual')
plt.scatter(x_test,y_pred,marker = 's',label='Predict')

plt.xlabel('Test Data')
plt.ylabel('Actual vs Predict')
plt.legend(loc='lower right')
plt.show()
