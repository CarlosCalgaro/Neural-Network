from lib.layers.layer import Layer

class Activation(Layer):

    def __init__(self, activation_function, activation_function_derivative) -> None:
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative

    def forward(self, inputs):
        self.inputs = inputs
        self.output = self.activation_function(self.inputs)
        return self.output
    
    def backward(self, output_error, learning_rate):
        return self.activation_function_derivative(self.inputs) * output_error