import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report,mean_squared_error
from sklearn.datasets import load_iris
model =load_iris()
feature = model.feature_names
target = model.target_names
print('Features:',feature,'\nTarget Names:',target)
df =
pd.DataFrame(np.c_[model['data'],model['target']],columns=model['feature_names']+['Species'])
feature = df.drop(['Species'],axis=1)
target = df['Species']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(feature,target,test_size=0.2,random_state=852)
from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)
y_pred1 = model.predict(X_test)
print(' ------------------------------------------------------ ','\nShowing report for rbf kernel model')
print(f'Confusion Matrix: \n{confusion_matrix(y_test,y_pred1)}')
print(f'Accuracy: {accuracy_score(y_test,y_pred1)}')
print(f"Precision: {precision_score(y_test,y_pred1,average='micro')}")
print(f"Recall: {recall_score(y_test,y_pred1,average='micro')}")
print(f"F1 Score: {f1_score(y_test,y_pred1,average='micro')}")
print(' ------------------------------------------------------ ')
from sklearn.svm import SVC
s = SVC(kernel='poly')
s.fit(X_train,y_train)
predict = s.predict(X_test)
print(' ------------------------------------------------------ ','\nShowing report for poly kernel model')
print(f'Confusion Matrix: \n{confusion_matrix(y_test,predict)}')
print(f'Accuracy: {accuracy_score(y_test,predict)}')
print(f"Precision: {precision_score(y_test,predict,average='micro')}")
print(f"Recall: {recall_score(y_test,predict,average='micro')}")
print(f"F1 Score: {f1_score(y_test,predict,average='micro')}")
print(' ------------------------------------------------------ ')
parameters = {
'C':[0.1,0.001,0.231,0.124,1.22],
'gamma':['scale','auto'],
'kernel':['poly','sigmoid','rbf'],
}
from sklearn.model_selection import GridSearchCV
grid_cv = GridSearchCV(model,parameters,cv=5)
search= grid_cv.fit(X_train,y_train)
search.best_params_
grid_model = SVC(C= 0.1, gamma = 'auto', kernel= 'poly')
grid_model.fit(X_train,y_train)
y_pred2 = grid_model.predict(X_test)
print(' ------------------------------------------------------ ','\nShowing report for Grid Search addedmodel')
print(f'Confusion Matrix: \n{confusion_matrix(y_test,y_pred2)}')
print(f'Accuracy: {accuracy_score(y_test,y_pred2)}')
print(f"Precision: {precision_score(y_test,y_pred2,average='micro')}")
print(f"Recall: {recall_score(y_test,y_pred2,average='micro')}")
print(f"F1 Score: {f1_score(y_test,y_pred2,average='micro')}")
print(' ------------------------------------------------------ ')
from sklearn.model_selection import RandomizedSearchCV
random_cv = RandomizedSearchCV(model,parameters,random_state=15)
random_search = random_cv.fit(X_train,y_train)
random_cv.best_params_
random_model = SVC(kernel='poly',gamma='scale',C=0.124)
random_model.fit(X_train,y_train)
y_pred3 = random_model.predict(X_test)
print(' ------------------------------------------------------ ','\nShowing report for Randomized Searchadded model')
print(f'Confusion Matrix: \n{confusion_matrix(y_test,y_pred3)}')
print(f'Accuracy: {accuracy_score(y_test,y_pred3)}')
print(f"Precision: {precision_score(y_test,y_pred3,average='micro')}")
print(f"Recall: {recall_score(y_test,y_pred3,average='micro')}")
print(f"F1 Score: {f1_score(y_test,y_pred3,average='micro')}")
print(' ------------------------------------------------------ ')
train = pd.read_csv('Training Data for Linear Regression.csv')
test = pd.read_csv('Testing Data for Linear Regression.csv')
X_train1 = train.drop(['Solar Radiation'],axis=1)
X_test1 = test.drop(['Solar Radiation'],axis=1)
y_train1 = train['Solar Radiation']
y_test1 = test['Solar Radiation']
from sklearn.svm import SVR
svm_regressor = SVR()
svm_regressor.fit(X_train1,y_train1)
predicted = svm_regressor.predict(X_test1)
print(f'Mean Square Error:{mean_squared_error(y_test1,predicted)}')
