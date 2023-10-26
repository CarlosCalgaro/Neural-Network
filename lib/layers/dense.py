
import numpy as np
from lib.layers.layer import Layer

class Dense(Layer):

    def __init__(self, n_inputs, n_neurons, activation_function = None) -> None:
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.1
        self.biases = np.zeros((1, n_neurons)) # Bias Vector using 0
        self.activation_function = activation_function

    def forward(self, inputs):
        self.inputs = inputs
        if len(self.inputs.shape) == 1:
            self.inputs = np.array([self.inputs])
        self.output = np.dot(self.inputs, self.weights) + self.biases
        if self.activation_function:
            self.output = self.activation_function(self.output)
        return self.output
    
    def backward(self, output_error, learning_rate = 0.2):
        self.dvalues = output_error
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.inputs.T, output_error)

        self.weights -=  learning_rate * weights_error
        self.biases -= learning_rate * output_error
        return input_error