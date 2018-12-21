"""
Solution for Machine Learning 2 HW4 E.6
Section 10 Monday
Designed By: Xiaochi (George) Li
"""

import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(42)


def draw(title, parameter, input_data):
    """Helper function Draw  --Draw dots and decision boundary
    :parameter
        title: string, the title of the plot
        parameter: dictionary, {"W": w, "b": b, }, contains W and b in numpy.matrix format
        input_data: list, [p1,p2,t], contains the point and it's target
    """
    plt.figure(figsize=(6, 6))
    training_set = []

    for i in input_data:
        training_set.append({"p": np.mat([[i[0]], [i[1]]]), "t": i[2]})
        if i[2] == 0:
            plt.plot(i[0], i[1], "bo")
        else:
            plt.plot(i[0], i[1], "ro")
    w = parameter["W"]
    b = parameter["b"]
    w1 = w[0, 0]
    w2 = w[0, 1]
    b = b[0, 0]
    # print(w1,w2,b)
    p1 = np.linspace(0, 5, 100)
    p2 = (w1 * p1 + b) / (-w2)
    plt.plot(p1, p2)
    plt.title(title)
    plt.show()


def check(input_data, parameter, echo):
    """Helper function check  -- Test whether the decision boundary classify the data points correctly
    :parameter
        input_data: list, [p1,p2,t], contains the point and it's target
        parameter: dictionary, {"W": w, "b": b, }, contains W and b in numpy.matrix format
        echo: boolean, whether prints the result of each check
    :returns
        boolean, whether all the data points have been correctly classified by decision boundary"""
    training_set = []

    for i in input_data:
        training_set.append({"p": np.mat([[i[0]], [i[1]]]), "t": i[2]})
    w = parameter["W"]
    b = parameter["b"]
    # print(w1,w2,b)
    for i in range(len(training_set)):  # check t and a for every point
        n = w * training_set[i]["p"] + b
        n = n.sum()

        if n < 0:
            a = 0
        else:
            a = 1

        if training_set[i]["t"] == a:
            if echo:
                print("vector:", i + 1)
                print(training_set[i]["p"].T)
                print("target:", training_set[i]["t"], "output:", a)
        else:
            return False
    return True


def perception(input_data, max_epoch):
    """function perception  -- single layer perception
    :parameter
        input_data: list, [p1,p2,t], contains the point and it's target
        max_epoch: integer, the maximum epoch
    :returns
        parameter: dictionary, {"W": w, "b": b, }, contains W and b in numpy.matrix format
    """
    training_set = []

    for i in input_data:
        training_set.append({"p": np.mat([[i[0]], [i[1]]]), "t": i[2]})
    w = np.mat([random.random(), random.random()])
    b = np.mat([random.random()])

    epoch = 0
    while epoch < max_epoch:
        print("Epoch:", epoch)
        print("W:", w)
        print("b:", b)
        draw("Epoch" + str(epoch), {"W": w, "b": b, }, input_data)
        epoch += 1

        for i in range(len(training_set)):
            n = w * training_set[i]["p"] + b
            n = n.sum()

            # a = hardlim(n)
            if n < 0:
                a = 0
            else:
                a = 1

            # update w and b
            if training_set[i]["t"] != a:
                w = w.T + (training_set[i]["t"] - a) * training_set[i]["p"]
                w = w.T
                b = b + (training_set[i]["t"] - a)

        # check after each iteration
        if check(input_data, {"W": w, "b": b, }, echo=False):
            draw("Epoch" + str(epoch), {"W": w, "b": b, }, input_data)
            return {"W": w, "b": b, }


input_data = [[1, 4, 0], [1, 5, 0], [2, 4, 0], [2, 5, 0], [3, 1, 1], [3, 2, 1], [4, 1, 1], [4, 2, 1]]
result = perception(input_data, max_epoch=100)
print("Result:", result)
print("Check Correctness:")
check(input_data, result, True)

