import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

x, y = np.mgrid[-5.0:15.0:30j, -10.0:10.0:30j]
# Need an (N, 2) array of (x, y) pairs.
xy = np.column_stack([x.flat, y.flat])

mu = np.array([2.0, 5.0])
sigma = np.array([[2.0, 0.0], [0.0, 2.0]])
covariance = np.diag(sigma**2)
z1 = multivariate_normal.pdf(xy, mean=mu, cov=covariance)/2

mu = np.array([8.0, 1.0])
sigma = np.array([[3.0, 1.0], [1.0, 3.0]])
covariance = np.diag(sigma**2)
z2 = multivariate_normal.pdf(xy, mean=mu, cov=covariance)

mu = np.array([5.0, 3.0])
sigma = np.array([[2.0, 1.0], [1.0, 2.0]])
covariance = np.diag(sigma**2)
z3 = multivariate_normal.pdf(xy, mean=mu, cov=covariance)/2


# Reshape back to a (30, 30) grid.
z1 = z1.reshape(x.shape)
z2 = z2.reshape(x.shape)
z3 = z3.reshape(x.shape)

ax = plt.subplot(111, projection='3d')
ax.scatter(x, y, z1)
ax.scatter(x, y, z2)
ax.scatter(x, y, z3)
plt.show()
