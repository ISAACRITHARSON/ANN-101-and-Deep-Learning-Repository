import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
train = pd.read_csv('Train.csv')
test = pd.read_csv('Solar Test.csv')
x_train = train.drop(['Solar Radiation'],axis=1)
y_train = train['Solar Radiation']
x_test = test.drop(['Solar Radiation'],axis=1)
y_test = test['Solar Radiation']
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver = 'lbfgs')
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
from sklearn.metrics import confusion_matrix
print(f’Confusion Matrix: {confusion_matrix(y_test,y_pred)}')
from sklearn.metrics import accuracy_score
print(f'Accuracy:{accuracy_score(y_test,y_pred)}')
from sklearn.metrics import recall_score
print(f'Recall:{recall_score(y_test,y_pred)}')
from sklearn.metrics import precision_score
print(f'Precision:{precision_score(y_test,y_pred)}')
