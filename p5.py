import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(100,3)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons) #no need for transpose now
        self.biases = np.zeros((1,n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
    def relu(self, inputs):
        
# print(X)

layer1= Layer_Dense(2,5)
layer2 = Layer_Dense(5,1)

layer1.forward(X)
layer1_output=layer1.output
layer2.forward(layer1_output)
print(layer2.output)

