import pandas as pd
import numpy as np
from lib.model import Model
from lib.layers.dense import Dense
from lib.layers.activation import Activation
from lib.activation_functions import *

data = pd.read_csv('./data/futebol.csv', sep=";")
data = data.drop(data.columns[0],axis=1)

def fromstring(data):
    return np.fromstring(data, sep=";")

result_types = {
    "H": 0,
    "D": 1,
    "A": 2
}

data["FTR"] = data["FTR"].apply(lambda x: np.eye(3)[result_types[x]])
data["HTR"] = data["HTR"].apply(lambda x: np.eye(3)[result_types[x]])

x_train = data.drop(columns=["FTAG", "FTHG", "HomeTeam", "AwayTeam", "FTR", "HTR"])

# Para um sistema preditivo de meio tempo pra frente, comentar abaixo:
x_train = x_train.drop(columns=["HTAG", "HTHG"])

y_train = data["FTR"]



model = Model()

model.sequential([
    Dense(12, 36),
    Activation(activation_function = ReLU, activation_function_derivative = ReLU_derivative),
    Dense(36, 48),
    Activation(activation_function = ReLU, activation_function_derivative = ReLU_derivative),
    Dense(48, 3),
    Activation(activation_function = sigmoid_stable, activation_function_derivative = sigmoid_derivative_stable),
])

# Normalizing data
# breakpoint()
print(x_train[:5])
x_train = (x_train - x_train.min())/(x_train.max() - x_train.min())
# x_train = x_train / np.sum(x_train, axis = 0)
# breakpoint()
# y_train = y_train / np.sum(y_train, axis = 0)
print(x_train[:5])

# Converting to numpy array

x_train = x_train.to_numpy()
y_train = y_train.to_numpy()

trim_index = int(len(x_train) * 0.8)
x_test = x_train[trim_index:]
x_train = x_train[:trim_index]

y_test =  y_train[trim_index:]
y_train = y_train[:trim_index]

model.fit(x_train, y_train, 300, 0.2)

correct_count = 0

for idx, inputs in enumerate(x_train):
    predicted = model.predict(inputs)
    result = np.eye(3)[np.argmax(predicted)]
    if np.array_equal(y_train[idx], result):
        correct_count += 1
    print("Predict ", inputs, y_train[idx], "=", result, "Original: ", predicted)

print(correct_count, " of ", len(x_train), " That's a ", (correct_count/len(x_train))*100, "% of correctness")

print("Now running against test data")
correct_count = 0
for idx, inputs in enumerate(x_test):
    predicted = model.predict(inputs)
    result = np.eye(3)[np.argmax(predicted)]
    if np.array_equal(y_test[idx], result):
        correct_count += 1
    # print("Predict ", inputs, y_train[idx], "=", result, "Original: ", predicted)
print(correct_count, " of ", len(x_test), " That's a ", (correct_count/len(x_test))*100, "% of correctness")
