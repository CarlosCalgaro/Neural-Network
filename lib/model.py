import numpy as np
from lib.layers.activation import Activation
from lib.optimizers.sgd import SGD
from lib.loss_functions import *
class Model:

    def __init__(self) -> None:
        self.layers = []

    def fit(self, x_train, y_train, epochs, learning_rate = 0.02, optimizer=SGD):
        samples = len(x_train)
        for i in range(epochs):
            # if i == 5:
                # breakpoint()
            # Forward pass
            err = 0
            for j in range(samples):
                output = np.array(x_train[j])
                # breakpoint()
                for layer in self.layers:
                    output = layer.forward(output)
            # Backward pass
                err += mse(y_train[j], output)
                error = mse_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)
                
            err /= samples
            print("Epoch ", i + 1, " of ", epochs, " %Err: ", err * 100, "%")

    def predict(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def predict_multiple(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)

        return result

    def sequential(self, layers = []):
        self.layers = layers

    def add_layer(self, layer):
        self.layers.append(layer)

    def inspect(self):
        for layer in self.layers:
            if(isinstance(layer, Activation)):
                print("Layer type: ", layer.__class__)           
            else:
                print("Layer type: ", layer.__class__ , "Input: ", layer.weights.shape[0] , "Output: ", layer.weights.shape[1])