import numpy as np

# initial values
ARRAY = []
with open("xoData.txt") as f:
    for line in f:
        ARRAY.append([int(x) for x in line.split()])

# step function (activation function)
def step_function(sum):
    if sum >= 0:
        return 1
    return -1


# calculateing output
def calculate_output(instance, weights, bias):
    sum = instance.dot(weights) + bias
    return step_function(sum)


# Hebbian Algorithm
def hebb():
    inputs = np.array(ARRAY)
    weights = np.array([0.0] * 25)
    bias = 0.0

    for i in range(len(inputs)):

        for j in range(len(inputs[i]) - 1):
            weights[j] = weights[j] + (inputs[i][j] * inputs[i][25])

        bias = bias + (1 * inputs[i][25])

    return weights, bias


# Perceptron Algorithm
def perceptron():
    inputs = np.array(ARRAY)
    weights = np.array([0.0] * 25)
    learning_rate = 0.1
    bias = 0.0

    x = 0
    while x < 100:

        x += 1
        for i in range(len(inputs)):
            s = inputs[i][:-1].dot(weights)
            prediction = step_function(s + bias)

            if inputs[i][25] != prediction:

                for j in range(len(inputs[i]) - 1):
                    weights[j] = weights[j] + (
                        learning_rate * inputs[i][j] * inputs[i][25]
                    )

                bias = bias + (learning_rate * inputs[i][25])

    return weights, bias


# Adaline Algorithm
def adaline():
    inputs = np.array(ARRAY)
    weights = np.array([0.0] * 25)
    learning_rate = 0.1
    bias = 0.0

    x = 0
    while x < 100:

        x += 1
        for i in range(len(inputs)):  
            s = inputs[i][:-1].dot(weights) + bias
            prediction = step_function(s)

            error = inputs[i][25] - s

            if inputs[i][25] != prediction:
                for j in range(len(inputs[i]) - 1):
                    weights[j] = weights[j] + (learning_rate * inputs[i][j] * error)

                bias = bias + (learning_rate * error)

    return weights, bias


# Multi Class Perceptron
def multiClassPerceptron():
    inputs = np.array(ARRAY)
    weights = np.array([[0.0] * 25, [0.0] * 25])
    learning_rate = 0.1
    bias = [0.0, 0.0]

    x = 0
    while x < 100:
        x += 1
        for i in range(len(inputs)):
            s1 = inputs[i][:-1].dot(weights[0])
            s2 = inputs[i][:-1].dot(weights[1])

            predictionX = step_function(s1 + bias[0])
            predictionO = step_function(s2 + bias[1])

            if inputs[i][25] != predictionX:

                for j in range(len(inputs[i]) - 1):
                    weights[0][j] = weights[0][j] + (
                        learning_rate * inputs[i][j] * inputs[i][25]
                    )

                bias[0] = bias[0] + (learning_rate * inputs[i][25])

            if (inputs[i][25] * (-1)) != predictionO:

                for j in range(len(inputs[i]) - 1):
                    weights[1][j] = weights[1][j] + (
                        learning_rate * inputs[i][j] * (inputs[i][25] * (-1))
                    )

                bias[1] = bias[1] + (learning_rate * (inputs[i][25] * (-1)))

    return weights, bias


# Multi Class Adaline
def multiClassAdaline():
    inputs = np.array(ARRAY)
    weights = np.array([[0.0] * 25, [0.0] * 25])
    learning_rate = 0.1
    bias = [0.0, 0.0]

    x = 0
    while x < 100:
        x += 1
        for i in range(len(inputs)):
            s1 = inputs[i][:-1].dot(weights[0]) + bias[0]
            s2 = inputs[i][:-1].dot(weights[1]) + bias[1]

            predictionX = step_function(s1)
            predictionO = step_function(s2)

            error1 = inputs[i][25] - s1
            error2 = (inputs[i][25] * (-1)) - s2

            if inputs[i][25] != predictionX:
                for j in range(len(inputs[i]) - 1):
                    weights[0][j] = weights[0][j] + (
                        learning_rate * inputs[i][j] * error1
                    )

                bias[0] = bias[0] + (learning_rate * error1)

            if (inputs[i][25] * (-1)) != predictionO:
                for j in range(len(inputs[i]) - 1):
                    weights[1][j] = weights[1][j] + (
                        learning_rate * inputs[i][j] * error2
                    )

                bias[1] = bias[1] + (learning_rate * error2)

    return weights, bias
