import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
train = pd.read_csv('Train.csv')
test = pd.read_csv('Solar Test.csv')
X_train = train[['Latitude', 'Longitude', 'Altitude', 'min Temo',

'Max Temp','Sunshine Hour']]
y_train = train['Solar Radiation']
X_test = test[['Latitude', 'Longitude', 'Altitude', 'min Temo',

'Max Temp','Sunshine Hour']]

y_test = test['Solar Radiation']
model = mlp(solver='lbfgs',max_iter=150)
model.fit(X_train,y_train)
parameter = {
'activation':['identity', 'logistic', 'tanh', 'relu'],
'solver':['lbfgs', 'sgd', 'adam'],
'alpha':[0.1,0.2,0.00003,0.11125,0.23232],
'hidden_layer_sizes':[(150,150),(250,250),(50,50),(25,25)]
}
from sklearn.model_selection import RandomizedSearchCV
random_cv = RandomizedSearchCV(model,parameter,random_state=125,cv=5)
random_model = random_cv.fit(X_train,y_train)
random_model.best_params_
parametrized_model =
mlp(solver='adam',alpha=0.2,hidden_layer_sizes=(150,150),activation='logistic')
parametrized_model.fit(X_train,y_train)
y_pred = parametrized_model.predict(X_test)
confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))
x=[]
for i in range(1,101):
model = mlp(max_iter=i+1,random_state=78)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
x.append(accuracy_score(y_test,y_pred))
plt.figure(figsize=(12,6))
for i in range(len(x)):
plt.plot(i,(1-x[i]),"ro-")
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('URK20AI1041')
plt.show()
