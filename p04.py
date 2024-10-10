
import numpy as np
np.random.seed(0)

#this is how batches work size==3
inputs = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8],
]
# layer 1 with 3 neuron
weights = [
    [0.2, 0.8, -0.5,1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

biases=[2, 3, 0.5]

#! layer 2 with 3 neuron, here the number of columns should be equal to the number of neuron in the last layer
weights2 = [
    [0.2, 0.8, -0.5],
    [0.5, -0.91, 0.26],
    [-0.26, -0.27, 0.17]
]

biases2=[2, 3, 0.5]

# Compute the output of layer 1
layer1_output = np.dot(inputs, np.array(weights).T) + biases
print("Layer 1 Output:")
print(layer1_output)

# Compute the output of layer 2
layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2
print("Layer 2 Output:")
print(layer2_output)


print("object oriented from now \n\n\n\n")
#! -------------------------------------------------------------------------------------------------------------

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8],
]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        #n_inputs is the number of features in a single entry
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons) #no need for transpose now
        self.biases = np.zeros((1,n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        

layer1 = Layer_Dense(4,5000000)
layer2 = Layer_Dense(5000000,1)

# Initialize the layers
layer1 = Layer_Dense(4, 5000000)
layer2 = Layer_Dense(5000000, 1)

# Forward pass through the first layer
layer1.forward(X)
# Forward pass through the second layer
layer2.forward(layer1.output)
print(f"Layer 1 Output Shape: {layer1.output.shape}")  # Display the shape of Layer 1's output


print(f"Layer 2 Output Shape: {layer2.output.shape}")  # Display the shape of Layer 2's output
print("Layer 2 Output Values:")
print(layer2.output)  # Display the actual output values of Layer 2
