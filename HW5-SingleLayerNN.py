import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)

p = np.linspace(-2, 2, 101)
t = np.exp(-np.abs(p)) * np.sin(np.pi * p)


def single_layer_nn(p, t, alpha=0.5, s=10, max_epoch=10):

    w1 = np.random.rand(s, 1)
    b1 = np.random.rand(s, 1)
    w2 = np.random.rand(1, s)
    b2 = np.random.rand(1, 1)
    for epoch in range(1, max_epoch + 1):

        F = 0
        a2_list = []
        for i in range(0,len(p)):
            pi = p[i]
            n1 = np.dot(w1, pi) + b1
            a1 = 1 / (1 + np.exp(-n1))
            a2 = np.dot(w2, a1) + b2
            a2_list.append(a2.flatten())

            F += (t[i] - a2) ** 2
            s2 = -2 * (t[i] - a2)
            df1 = (1 - a1) * a1
            s1 = np.dot(np.diag(df1.flatten()), w2.T) * s2

            w2 = w2 - alpha * np.dot(s2, a1.T)
            b2 = b2 - alpha * s2
            w1 = w1 - alpha * np.dot(s1, p[i])
            b1 = b1 - alpha * s1
            parameters = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
        print("Epoch:", epoch, "MSE:", F.squeeze())

    return parameters, a2_list


parameter, a2_list = single_layer_nn(p, t, s=2, alpha=1)
prediction = np.array(a2_list).reshape(-1, 1)
plt.plot(p[1:], t[1:], label="Actual")
plt.plot(p[1:], prediction[1:], label="NN Prediction")
plt.legend()
plt.title("s1=2, alpha=1")
plt.show()

parameter, a2_list = single_layer_nn(p, t, s=2, alpha=0.5)
prediction = np.array(a2_list).reshape(-1, 1)
plt.plot(p[1:], t[1:], label="Actual")
plt.plot(p[1:], prediction[1:], label="NN Prediction")
plt.legend()
plt.title("s1=2, alpha=0.5")
plt.show()

parameter, a2_list = single_layer_nn(p, t, s=2, alpha=0.1)
prediction = np.array(a2_list).reshape(-1, 1)
plt.plot(p[1:], t[1:], label="Actual")
plt.plot(p[1:], prediction[1:], label="NN Prediction")
plt.legend()
plt.title("s1=2 alpha = 0.1")
plt.show()

parameter, a2_list = single_layer_nn(p, t, s=10, alpha=1)
prediction = np.array(a2_list).reshape(-1, 1)
plt.plot(p[1:], t[1:], label="Actual")
plt.plot(p[1:], prediction[1:], label="NN Prediction")
plt.legend()
plt.title("s1=10, alpha=1")
plt.show()

parameter, a2_list = single_layer_nn(p, t, s=10, alpha=0.5)
prediction = np.array(a2_list).reshape(-1, 1)
plt.plot(p[1:], t[1:], label="Actual")
plt.plot(p[1:], prediction[1:], label="NN Prediction")
plt.legend()
plt.title("s1=10, alpha=0.5")
plt.show()

parameter, a2_list = single_layer_nn(p, t, s=10, alpha=0.1)
prediction = np.array(a2_list).reshape(-1, 1)
plt.plot(p[1:], t[1:], label="Actual")
plt.plot(p[1:], prediction[1:], label="NN Prediction")
plt.legend()
plt.title("s1=10 alpha=0.1")
plt.show()