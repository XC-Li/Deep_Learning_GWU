import numpy as np
import matplotlib.pyplot as plt


p = np.linspace(-2, 2, 100).reshape(1,-1)
t = np.exp(-np.abs(p)) * np.sin(np.pi * p)
print(p.shape)

max_epoch = 10
alpha = 0.5
s = 10

w1 = np.random.rand(s, 1)
b1 = np.random.rand(s, 1)
w2 = np.random.rand(1, s)
b2 = np.random.rand(1, 1)


#for n_epoch in range(max_epoch):
n1 = np.dot(w1, p) + b1
a1 = 1 / (1 + np.exp(-n1))
a2 = np.dot(w2, a1) + b2
e = t - a2
s2 = -2 * (t-a2)
df1 = (1-a1) * a1
print(df1.shape)
print(w2.shape)
print(s2.shape)
s1 = np.dot(w2, df1) * s2
