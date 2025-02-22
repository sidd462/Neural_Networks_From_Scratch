import numpy as np


softmax_output = [0.7,0.1,0.2]
target_output = [1,0,0]
target_class = 0

#! Categorical cross entropy loss
# loss = -(math.log(softmax_output[0])*target_output[0] + 
#         (math.log(softmax_output[1])*target_output[1]) + 
#         math.log(softmax_output[2])*target_output[2])
loss = -np.sum(target_output * np.log(softmax_output))

print(loss)