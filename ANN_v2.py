#!/usr/bin/env python
# coding: utf-8

# In[163]:


import numpy as np
#Independent variables
input_set = np.array([[[1,1]],[[7,2]],[[3,3]],[[8,3]],[[2,9]],[[10,3]],[[5,3]],[[11,2]],[[3,1]],[[7,5]],[[8,1]],[[6,2]]])#Dependent variable
labels = np.array([0,1, 0,1,0,1,0,1,0,1,1,0])
labels = labels.reshape(12,1) #to convert labels to vector


# In[182]:


class Layer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) / np.sqrt(input_size + output_size)
        self.bias = np.random.randn(1, output_size) / np.sqrt(input_size + output_size)

    def forward(self, input):
        self.input = input
        print ("s")
        return np.dot(input, self.weights) + self.bias

    def backward(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # bias_error = output_error
        
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error


# In[124]:



def sigmoid(x):
    return 1/(1+np.exp(-x))


# In[125]:



def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))


# In[126]:


def mse(labels, y_pred):
    return np.mean(np.power(labels - y_pred, 2))


# In[137]:


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_pred.size


# In[170]:


learning_rate = 0.005                 

# Layer obj
layer1 = Layer(n_inputs, n_neurons)
layer2 = Layer(n_neurons, n_neurons)

for i in range(100):
    error=0  
    for j in range  (len((input_set))):
        output=input_set[j] # feedforward input
        z = layer1.forward(output)
        error += mse(labels[j], z)
        output_error = mse_prime(labels[j],  z)
        output_error = layer1.backward(output_error, learning_rate)
    error /= len(input_set)
    print('%d/%d, error=%f' % (i + 1, epochs, error))


# In[200]:


def predict( input):
    output = input
    output = layer1.forward(output)
    return output


# In[ ]:


ratio = sum([np.argmax(y) == np.argmax(predict(network, x)) for x, y in zip(x_test, y_test)]) / len(x_test)
error = sum([mse(y, predict( x)) 
print('ratio: %.2f' % ratio)
print('mse: %.4f' % error)


# In[203]:


error =  predict([[7,2]])
result = error.mean()


# In[204]:


import math
print(result)
if result >0.5:
    f=math.ceil(result)
    print(f)
else:
    f=math.floor(result)
    print(f)
if f==0:
    print("this point is in class 1")
else:
    print ("this point is in class 2")


# In[ ]:




