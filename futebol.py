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

input_data = data.drop(columns=["FTAG", "FTHG", "HomeTeam", "AwayTeam", "FTR", "HTR"])

# Para um sistema preditivo de meio tempo pra frente, comentar abaixo:
input_data = input_data.drop(columns=["HTAG", "HTHG"])

output_data = data["FTR"]

print(input_data.head())

print(output_data)


model = Model()

model.sequential([
    # Dense(14, 36),
    Dense(12, 36),
    Activation(activation_function = sigmoid_stable, activation_function_derivative = sigmoid_derivative_stable),
    Dense(36, 48),
    Activation(activation_function = sigmoid_stable, activation_function_derivative = sigmoid_derivative_stable),
    # Dense(48, 123),
    # Activation(activation_function = sigmoid, activation_function_derivative = sigmoid_derivative),
    Dense(48, 3),
    Activation(activation_function = sigmoid_stable, activation_function_derivative = sigmoid_derivative_stable),
    # Activation(activation_function = softmax, activation_function_derivative = softmax_derivative),
])


print(input_data[:5])
model.fit(input_data.to_numpy(), output_data.to_numpy(), 200)

correct_count = 0
for idx, inputs in enumerate(input_data.to_numpy()):
    predicted = model.predict(inputs)
    predicted = np.eye(3)[np.argmax(predicted)]
    if np.array_equal(output_data[idx], predicted):
        correct_count += 1
    print("Predict ", inputs, output_data[idx], "=", predicted)

print(correct_count, " of ", len(input_data), " That's a ", (correct_count/len(input_data))*100, "% of correctness")