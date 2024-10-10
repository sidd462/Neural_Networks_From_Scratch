import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

X, y = spiral_data(100,3)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        #n_inputs is the number of features in a single entry
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons) #no need for transpose now
        self.biases = np.zeros((1,n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)

class Activation_Softmax:
    def forward(self, inputs):
        self.output = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        exp=np.sum(self.output,axis=1, keepdims=True)
        self.output= self.output/exp

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_categoricalcrossentropy(Loss):
    def forward(self, y_pred,y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
        

layer1= Layer_Dense(2,3)
activate1 = Activation_Softmax()
layer1.forward(X)
activate1.forward(layer1.output)
print(activate1.output[:5])

loss_function = Loss_categoricalcrossentropy()
loss = loss_function.calculate(activate1.output,y)
print(loss)

