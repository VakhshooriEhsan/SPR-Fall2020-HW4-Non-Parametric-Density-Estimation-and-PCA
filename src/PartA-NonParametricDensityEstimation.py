import numpy as np
import matplotlib.pyplot as plt

def K(u):
    for i in range(len(u)):
        if(abs(u[i])>=0.5):
            return 0
    return 1

def KDE(x, y, h, N, D):
    p_KDE = np.zeros(N)
    k=0
    for i in range(N):
        k = 0
        for j in range(N):
            k += K([(x[i]-x[j])/h, (y[i]-y[j])/h])
        p_KDE[i] = 1/(N*h1**D)*k
    return p_KDE


N = 500 # number of examples
D = 2 # number of dimensions

mean = [2, 5]
cov = [[2, 0], [0, 2]]
x1, y1 = np.random.multivariate_normal(mean, cov, N).T

mean = [8, 1]
cov = [[3, 1], [1, 3]]
x2, y2 = np.random.multivariate_normal(mean, cov, N).T

mean = [5, 3]
cov = [[2, 1], [1, 2]]
x3, y3 = np.random.multivariate_normal(mean, cov, N).T

h1 = 0.09
h2 = 0.3
h3 = 0.6

p_KDE = KDE(x1, y1, h3, N, D)

plt.figure(1)
plt.plot(x1, y1, '.')
plt.plot(x2, y2, '.')
plt.plot(x3, y3, '.')

plt.figure(2)
ax = plt.subplot(111, projection='3d')
ax.scatter(x1, y1, p_KDE, '.')
ax.scatter(x2, y2, p_KDE, '.')
ax.scatter(x3, y3, p_KDE, '.')

plt.figure(3)
plt.hist(p_KDE, bins=25, alpha=0.7)
plt.hist(x2, bins=25, alpha=0.7)
plt.hist(x3, bins=25, alpha=0.7)
plt.show()
