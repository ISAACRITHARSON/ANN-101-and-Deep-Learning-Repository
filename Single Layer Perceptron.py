import pandas as pd
import matplotlib.pyplot as plt
path = 'Train.csv'
df = pd.read_csv(path)
#df.info()
X1 = df['Latitude'].tolist()
X2 = df['Longitude'].tolist()
X3 = df['Altitude'].tolist()
X4 = df['min Temo'].tolist()
X5 = df['Max Temp'].tolist()
X6 = df['Sunshine Hour'].tolist()
D1 = df['Solar Radiation'].tolist()
xmin1,xmax1 = min(X1),max(X1)
xmin2,xmax2 = min(X2),max(X2)
xmin3,xmax3 = min(X3),max(X3)
xmin4,xmax4 = min(X4),max(X4)
xmin5,xmax5 = min(X5),max(X5)
xmin6,xmax6 = min(X6),max(X6)
for i in range(len(X1)):
X1[i] = X1[i] - xmin1
X1[i] = X1[i]/(xmax1 - xmin1)
X2[i] = X2[i] - xmin2
X2[i] = X2[i]/(xmax2 - xmin2)
X3[i] = X3[i] - xmin3
X3[i] = X3[i]/(xmax3 - xmin3)
X4[i] = X4[i] - xmin4
X4[i] = X4[i]/(xmax4 - xmin4)
X5[i] = X5[i] - xmin5
X5[i] = X5[i]/(xmax5 - xmin5)
X6[i] = X6[i] - xmin6
X6[i] = X6[i]/(xmax6 - xmin6)
accuracy_list = []
z = 0
w1 = 0.1
w2 = 0.2
w3 = 0.3
w4 = -0.7
w5 = -1.2
w6 = 0.25
correct = 0
c = 0.1
A1 = []
f = {}
for z in range(100):
for i in range(0,len(X1)):
x1 = X1[i]
x2 = X2[i]
x3 = X3[i]
x4 = X4[i]
x5 = X5[i]
x6 = X6[i]
A1.append(x1 * w1 + x2 * w2 + x3 * w3 + x4 * w4 + x5 * w5 + x6 * w6)
#print(i)
#print(A1[i])
if A1[i] >= 0.5:
A1[i] = 1
else:
A1[i] = 0
#print("Predicted: ",A1[i])
#print("Actual: ",D1[i])
if A1[i] == D1[i]:
error1 = 0
else:
error1 = D1[i] - A1[i]
#print("error = ",error1)
if error1 != 0:
  w1 = round(w1 + c*error1*x1,2)
w2 = round(w2 + c*error1*x2,2)
w3 = round(w3 + c*error1*x3,2)
w4 = round(w4 + c*error1*x4,2)
w5 = round(w5 + c*error1*x5,2)
w6 = round(w6 + c*error1*x6,2)
if A1[i] == D1[i]:
correct +=1
#print(w1,w2,w3,w4,w5,w6)
if z !=100:
A1 = []
acc = correct/len(D1)
accuracy_list.append(acc)
f.update({z: acc})
#print("Accuracy",acc)
correct = 0
else:
print(acc)
f.update({z: acc})
accuracy_list.append(acc)
print(w1,w2,w3,w4,w5,w6)
print(acc)
Keymax = max(zip(f.values(), f.keys()))[1]
print(Keymax)
plt.figure(figsize=(12,6))
for i in range(len(accuracy_list)):
plt.plot(i,accuracy_list[i],"ro-")
plt.show()
path2 = 'Solar Test.csv'
df1 = pd.read_csv(path2)
#df.info()
A1 = df1['Latitude'].tolist()
A2 = df1['Longitude'].tolist()
A3 = df1['Altitude'].tolist()
A4 = df1['min Temo'].tolist()
A5 = df1['Max Temp'].tolist()
A6 = df1['Sunshine Hour'].tolist()
D0 = df1['Solar Radiation'].tolist()
size1 = len(A1)
print(len(D0))
min1 = min(A1)
max1 = max(A1)
#print(xmax1,xmin1)
min2 = min(A2)
max2 = max(A2)
#print(xmax2,xmin2)
min3 = min(A3)
max3 = max(A3)
#print(xmax3,xmin3)
min4 = min(A4)
max4 = max(A4)
#print(xmax4,xmin4)
min5 = min(A5)
max5 = max(A5)
#print(xmax5,xmin5)
min6 = min(A6)
max6 = max(A6)
#print(xmax6,xmin6)
for i in range(size1):
A1[i] = A1[i] - min1
A1[i] = A1[i]/(max1 - min1)
#print(X1[i])
print(len(A1))
for i in range(size1):
  A2[i] = A2[i] - min2
A2[i] = A2[i]/(max2 - min2)
#print(X1[i])
print(len(A2))
for i in range(size1):
A3[i] = A3[i] - min3
A3[i] = A3[i]/(max3 - min3)
#print(X1[i])
print(len(A3))
for i in range(size1):
A4[i] = A4[i] - min4
A4[i] = A4[i]/(max4 - min4)
#print(X1[i])
print(len(A4))
for i in range(size1):
A5[i] = A5[i] - min5
A5[i] = A5[i]/(max5 - min5)
#print(X1[i])
print(len(A5))
for i in range(size1):
A6[i] = A6[i] - min6
A6[i] = A6[i]/(max6 - min6)
#print(X1[i])
print(len(A6))
W1,W2,W3,W4,W5,W6 = w1,w2,w3,w4,w5,w6
predicted_correctly = 0
length = len(A1)
A0 = []
for i in range(length):
input1 = A1[i]
input2 = A2[i]
input3 = A3[i]
input4 = A4[i]
input5 = A5[i]
input6 = A6[i]
A0.append(round(input1 * W1 + input2 * W2 + input3 * W3 + input4 * W4 + input5 * W5
+ input6 * W6,2))
#print("i: ",i," Predicted: ",A0[i])
#print("Actual: ",D0[i])
if A0[i] >= 0.46:
A0[i] = 1
else:
A0[i] = 0
#if A0[i] != D0[i]:
#print(i)
#print("Predicted: ",A0[i])
#print("Actual: ",D0[i])
if A0[i] == D0[i]:
predicted_correctly +=1
accuracy = predicted_correctly/len(D0)
print("Accuracy: ",accuracy)
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(D0, A0))
print(classification_report(D0, A0))
