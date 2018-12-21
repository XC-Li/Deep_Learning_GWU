# Machine Learning 2 Home work 3
# Section 10 Monday
# Author: Xiaochi (George) Li

import numpy as np
import matplotlib.pyplot as plt


# Exercise 1
def poslin(mat):
    new_mat = np.empty(mat.shape)
    for row in range(mat.shape[0]):
        for col in range(mat.shape[1]):
            if mat[row, col] < 0:
                new_mat[row, col] = 0
            else:
                new_mat[row, col] = mat[row, col]
    return new_mat


def purelin(x):
    return x


def nn1(p):
    W1 = np.array([[-1], [1]])
    b1 = np.array([[0.5], [1]])
    W2 = np.array([1, 1]).reshape(1,2)
    b2 = [-1]
    a1 = poslin(np.dot(W1, p) + b1)
    a2 = purelin(np.dot(W2, a1) + b2)
    return a2[0,0]


a2 = [nn1(p) for p in np.linspace(-2., 2., 100)]
plt.plot(np.linspace(-2., 2., 100),a2)
plt.title("Exercise 1")
plt.show()


# Exercise 2


def hardlims(x):
    if x < 0:
        return -1
    else:
        return 1


def hardlim(x):
    if x < 0:
        return 0
    else:
        return 1


def satlins(x):
    if x < -1:
        return -1
    elif x > 1:
        return 1
    else:
        return x

prange = np.linspace(-2., 2., 100)
# 1
y1 = [hardlims(1 * p + 1) for p in prange]
plt.plot(prange, y1)
plt.title("Exercise 2-1")
plt.show()

#2
y2 = [hardlim(1 * p + 1) for p in prange]
plt.title("Exercise 2-2")
plt.plot(prange, y2)
plt.show()

#3
y3 = [purelin(2 * p + 3) for p in prange]
plt.title("Exercise 2-3")
plt.plot(prange, y3)
plt.show()

#4
y4 = [satlins(2 * p + 3) for p in prange]
plt.title("Exercise 2-4")
plt.plot(prange, y4)
plt.show()

#5
y5 = [max(2 * p + 1, 0) for p in prange]
plt.title("Exercise 2-5")
plt.plot(prange, y5)
plt.show()


# Exercise 3
def satlins(mat):
    new_mat = np.empty(mat.shape)
    for row in range(mat.shape[0]):
        for col in range(mat.shape[1]):
            if mat[row, col] < -1:
                new_mat[row, col] = -1
            elif mat[row, col] > 1:
                new_mat[row, col] = 1
            else:
                new_mat[row, col] = mat[row, col]
    return new_mat


W1 = np.array([[2], [1]])
b1 = np.array([[2], [-1]])
W2 = np.array([1, -1])
b2 = 0

p = np.linspace(-3, 3, 100).reshape(1,-1)
n1 = np.dot(W1, p) + b1
n1_1 = n1[0]
n1_2 = n1[1]
a1 = satlins(n1)
a1_1 = a1[0]
a1_2 = a1[1]
n2 = np.dot(W2, a1) + b2

plt.title("Exercise 3-1: n1_1")
plt.plot(np.linspace(-3, 3, 100),n1_1)
plt.show()
plt.title("Exercise 3-2 a1_1")
plt.plot(np.linspace(-3, 3, 100),a1_1)
plt.show()
plt.title("Exercise 3-3 n1_2")
plt.plot(np.linspace(-3, 3, 100),n1_2)
plt.show()
plt.title("Exercise 3-4 a1_2")
plt.plot(np.linspace(-3, 3, 100),a1_2)
plt.show()
plt.title("Exercise 3-5 n2_1 same as Exercise 3-6 a2_1")
plt.plot(np.linspace(-3, 3, 100),n2)
plt.show()