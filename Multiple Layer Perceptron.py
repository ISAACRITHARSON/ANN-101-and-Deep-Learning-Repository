import math
import numpy as np
import matplotlib.pyplot as plt
# These are XOR inputs
x=np.array([[0,0,1,1],[0,1,0,1]])
# These are XOR outputs
y=np.array([[0,1,1,0]])
# Number of inputs
num_of_x = 2
# Number of nodes in the output layer
num_of_output_nodes = 1
# Number of nodes in the hidden layer
num_of_hidden_nodes = 2
# Total training examples
m = x.shape[1]
# Learning rate
c = 0.1
# Define random seed for consistent results
np.random.seed(2)
# Define weight matrices for neural network
w1 = np.random.rand(num_of_hidden_nodes,num_of_x) # Weight matrix for hidden layer
w2 = np.random.rand(num_of_output_nodes,num_of_hidden_nodes) # Weight matrix for
output layer
# We will use this list to accumulate losses
losses = []
print(w1)
print()
print(w2)
#sigmoid activation function for hidden layer and output
def sigmoid(z):
z= 1/(1+np.exp(-z))
return z
# Forward propagation
def forward_prop(w1,w2,x):
z1 = np.dot(w1,x) # weight * input
a1 = sigmoid(z1) # activation of summation
z2 = np.dot(w2,a1) # weight * activation
a2 = sigmoid(z2) # activation of output
return z1,a1,z2,a2
# Backward propagation
def back_prop(m,w1,w2,z1,a1,z2,a2,y):

dz2 = a2-y # activation - actual
dw2 = np.dot(dz2,a1.T)/m # output node weight update : error * activation / num_of_samples
dz1 = np.dot(w2.T,dz2) * a1*(1-a1) # hidden node error: error * old weight * activation * (1 -
activation)
dw1 = np.dot(dz1,x.T)/m # hidden node weight update : error * activation / num_of_samples

dw1 = np.reshape(dw1,w1.shape)
dw2 = np.reshape(dw2,w2.shape)
return dz2,dw2,dz1,dw1
iterations = 100000
for i in range(iterations):
z1,a1,z2,a2 = forward_prop(w1,w2,x)
loss = -(1/m)*np.sum(y*np.log(a2)+(1-y)*np.log(1-a2))
losses.append(loss)
da2,dw2,dz1,dw1 = back_prop(m,w1,w2,z1,a1,z2,a2,y)
w2 = w2-c*dw2
w1 = w1-c*dw1

# We plot losses to see how our network is doing
plt.plot(losses)
plt.xlabel("EPOCHS")
plt.ylabel("Loss value")
def predict(w1,w2,input):
z1,a1,z2,a2 = forward_prop(w1,w2,test)
a2 = np.squeeze(a2)
if a2>=0.5:
print("For input", [i[0] for i in input], "output is 1")
else:
print("For input", [i[0] for i in input], "output is 0")
test = np.array([[0],[0]])
predict(w1,w2,test)
test = np.array([[0],[1]])
predict(w1,w2,test)
test = np.array([[1],[0]])
predict(w1,w2,test)
test = np.array([[1],[1]])
predict(w1,w2,test)
