import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import math
nnfs.init()

X, y = spiral_data(100,3)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        #n_inputs is the number of features in a single entry
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons) #no need for transpose now
        self.biases = np.zeros((1,n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_Softmax:
    def forward(self, inputs):
        self.output = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        exp=np.sum(self.output,axis=1, keepdims=True)
        self.output= self.output/exp
        
# print(X)

layer1= Layer_Dense(2,3)
activate1 = Activation_Softmax()
layer1.forward(X)
activate1.forward(layer1.output)
print(activate1.output[:5])


