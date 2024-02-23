import numpy as np
import matplotlib.pyplot as plt

xy = np.array([[-0.95, +0.02],
               [-0.82, +0.03],
               [-0.62, -0.17],
               [-0.43, -0.12],
               [-0.17, -0.37],
               [-0.07, -0.25],
               [+0.25, -0.10], 
               [+0.38, +0.14],
               [+0.61, +0.53],
               [+0.79, +0.71],
               [+1.04, +1.53]])

nb_iteration = 1000
mu = 0.003
Ls = np.zeros(nb_iteration)

N = 3
a = np.zeros((N,), dtype=np.float32)

for n in range(nb_iteration):
    L = 0.0
    grad = np.zeros(a.shape, dtype=np.float32)
    
    for i in range(0, xy.shape[0]):
        xi = xy[i, 0]
        yi = xy[i, 1]

        x = xi ** np.arange(0, N)
        yhati = np.dot(a, x)
        Li = (yi - yhati) ** 2 
        gradi = 2*(yhati - yi) * x

        L += Li
        grad += gradi
    
    a = a - mu * grad
    Ls[n] = L

plt.plot(Ls)
plt.show()


xpoints = np.linspace(-1.25, 1.25, 101)

ypoints = 0 
for n in range(N):
    ypoints += a[n] * xpoints ** n

plt.scatter(xy[:, 0], xy[:, 1])
plt.plot(xpoints, ypoints)
plt.show()
