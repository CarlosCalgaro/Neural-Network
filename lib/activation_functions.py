import numpy as np

def step_function(inputs):
    return 1 if (inputs > 0) else 0

def sigmoid(inputs):
    return 1/(1 + np.exp(-inputs))
def sigmoid_stable(x):
    return np.where(
            x >= 0, # condition
            1 / (1 + np.exp(-x)), # For positive values
            np.exp(x) / (1 + np.exp(x)) # For negative values
    )

def sigmoid_derivative_stable(x):
    sigmoid_of_x = sigmoid_stable(x)
    return sigmoid_of_x * (1 - sigmoid_of_x)

def sigmoid_derivative(x):
    return np.exp(-x) / ((1 + np.exp(-x))**2)

def ReLU(inputs):
    return np.maximum(0, inputs)

def ReLU_derivative(inputs):
     return (inputs>0)*np.ones(inputs.shape)

def tanh(inputs):
    return np.tanh(inputs)

def tanh_derivative(inputs):
    return 1-np.tanh(inputs)**2

def softmax(inputs):
    val = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    return val / np.sum(val, axis=1, keepdims=True)

