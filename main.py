import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from lib.layers.dense import Dense
from lib.layers.activation import Activation
from lib.activation_functions import *
from lib.loss.categorical_crossentropy import CategoricalCrossentropy
nnfs.init()

X, y = nnfs.datasets.spiral_data(samples=100, classes=3)

dense1 = Dense(2, 3)
activation1 = Activation(activation_function=ReLU)

dense2 = Dense(3, 3)
activation2 = Activation(activation_function=softmax)

loss_function = CategoricalCrossentropy()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)


print(activation2.output[:5])

# Perform a forward pass through loss function
# it takes the output of second dense layer here and returns loss
loss = loss_function.calculate(activation2.output, y)
# Print loss value
print('loss:', loss)

predictions = np.argmax(activation2.output, axis=1)
if len(y.shape) == 2:
 y = np.argmax(y, axis=1)
accuracy = np.mean(predictions==y)
# Print accuracy
print('acc:', accuracy) 