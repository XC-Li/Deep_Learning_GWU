import numpy as np
import matplotlib.pyplot as plt
import random
import math
from sympy import *

# functions used


def logsig(x):
    y = 1 / (1 + 1 / math.e ** x)
    return y


def purelin(x):
    y = x
    return y


def g(x):
    y = 1/math.e**(abs(x)) * math.sin(math.pi * x)
    return y


# ---------------------- S=2 ----------------------------------------
"""
trainingSet = np.arange(-2, 2, 0.4)
outputs = []
targets = []
mse = []
S = 2   # 1-S-1 network
W1 = np.random.uniform(-0.5, 0.5, S)
b1 = np.random.uniform(-0.5, 0.5, S)
W2 = np.random.uniform(-0.5, 0.5, S)
b2 = np.random.random(1)
alpha = 0.1
epochs = 50

# propagate
for epoch in range(epochs):
    error = np.zeros(len(trainingSet))
    for i in range(len(trainingSet)):
        a0 = trainingSet[i]
        n1 = W1 * a0 + b1
        a1 = logsig(n1)
        n2 = np.dot(W2, a1) + b2
        a2 = purelin(n2)

        # calculate error
        e = (g(a0) - a2)

        # back propagate
        x = Symbol("x")
        f2_diff = diff(purelin(x), x)
        s2 = -2 * f2_diff * e
        f1_diff = (1 - a1) * a1
        s1 = np.diagflat(f1_diff).dot(W2.transpose()) * s2

        # update weights and bias
        W2 = W2 - alpha * s2 * a1.transpose()
        b2 = b2 - alpha * s2
        W1 = W1 - alpha * s1 * a0
        b1 = b1 - alpha * s1
        error[i] = abs(e)
    mse.append(error.transpose().dot(error))
print("E2: If S1=2, the weights and bias of trained perceptron is:", "output layer:", W2, b2, "hidden layer:", W1, b1)

# plot mse
fig = plt.figure()
x = np.arange(0, epochs, 1)
y = mse
plt.xlabel("epochs", fontsize=18)
plt.ylabel("mse", fontsize=18)
plt.plot(x, y)
plt.show()

# test and plot original function and the approximation
testSet = np.arange(-2, 2, 0.1)
for step in range(len(testSet)):
    a0 = testSet[step]
    n1 = W1 * a0 + b1
    a1 = logsig(n1)
    n2 = np.dot(W2, a1) + b2
    a2 = purelin(n2)
    outputs.append(a2)
    targets.append(g(a0))

figure = plt.figure()
plt.plot(testSet, outputs, label="approximation")
plt.plot(testSet, targets, 'r-.', label="1+sin(x*π/2)")
plt.legend()

plt.show()
"""
# -------------------------- S=10 -----------------------------------
trainingSet = np.arange(-2, 2, 0.4)
outputs = []
targets = []
mse = []
S = 10  # 1-S-1 network
W1 = np.random.uniform(-0.5, 0.5, S)
b1 = np.random.uniform(-0.5, 0.5, S)
W2 = np.random.uniform(-0.5, 0.5, S)
b2 = np.random.random(1)
alpha = 0.1
epochs = 50

# propagate
for epoch in range(epochs):
    error = np.zeros(len(trainingSet))
    for i in range(len(trainingSet)):
        a0 = trainingSet[i]
        n1 = W1 * a0 + b1
        a1 = logsig(n1)
        n2 = np.dot(W2, a1) + b2
        a2 = purelin(n2)

        # calculate error
        e = (g(a0) - a2)

        # back propagate
        x = Symbol("x")
        f2_diff = diff(purelin(x), x)
        s2 = -2 * f2_diff * e
        f1_diff = (1 - a1) * a1
        s1 = np.diagflat(f1_diff).dot(W2.transpose()) * s2

        # update weights and bias
        W2 = W2 - alpha * s2 * a1.transpose()
        b2 = b2 - alpha * s2
        W1 = W1 - alpha * s1 * a0
        b1 = b1 - alpha * s1
        error[i] = abs(e)
    mse.append(error.transpose().dot(error))


# plot mse
fig = plt.figure()
x = np.arange(0, epochs, 1)
y = mse
plt.xlabel("epochs", fontsize=18)
plt.ylabel("mse", fontsize=18)
plt.plot(x, y)
plt.show()

# test and plot original function and the approximation
testSet = np.arange(-2, 2, 0.1)
for step in range(len(testSet)):
    a0 = testSet[step]
    n1 = W1 * a0 + b1
    a1 = logsig(n1)
    n2 = np.dot(W2, a1) + b2
    a2 = purelin(n2)
    outputs.append(a2)
    targets.append(g(a0))

figure = plt.figure()
plt.plot(testSet, outputs, label="approximation")
plt.plot(testSet, targets, 'r-.', label="original")
plt.legend()

plt.show()
