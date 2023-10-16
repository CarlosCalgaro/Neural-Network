
class SGD:
    def __init__(self, learning_rate = 1.0) -> None:
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        bias_change = -self.learning_rate * layer.dvalues
        print("lr: ", self.learning_rate, " dvalue: ", layer.dvalues, "result: ", bias_change)
        layer.biases += bias_change
