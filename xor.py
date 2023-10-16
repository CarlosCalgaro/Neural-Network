from lib.model import Model
from lib.layers.dense import Dense
from lib.layers.activation import Activation

from lib.activation_functions import *

x_train = np.random.randn(200)
y_train = np.sin(x_train)

x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

# x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
# y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

model = Model()
model.sequential([
    Dense(2, 3),
    Activation(activation_function = tanh, activation_function_derivative = tanh_derivative),
    Dense(3, 1),
    Activation(activation_function = tanh, activation_function_derivative = tanh_derivative)
])


model.inspect()
model.fit(x_train, y_train, 100000, 0.2)

# print(model.predict([[0, 1]]))
for inputs in x_train:
    predicted = model.predict(inputs)
    print("Predict ", inputs, "=", predicted)
