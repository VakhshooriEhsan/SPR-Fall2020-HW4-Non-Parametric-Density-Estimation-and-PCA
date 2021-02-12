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

def Histogram(x, y, X, Y):
    x = np.concatenate((x, [2*x[len(x)-1]-x[len(x)-2]]), axis=0)
    y = np.concatenate((y, [2*y[len(y)-1]-y[len(y)-2]]), axis=0)
    z, x1, y1 = np.histogram2d(X, Y, bins=(x, y))
    return z

def plting(X, Y, h, fun):
    x_max = np.max(X[0])
    x_min = np.min(X[0])
    y_max = np.max(Y[0])
    y_min = np.min(Y[0])
    for i in range(len(X)):
        x_max = max(np.max(X[i]), x_max)
        x_min = min(np.min(X[i]), x_min)
        y_max = max(np.max(Y[i]), y_max)
        y_min = min(np.min(Y[i]), y_min)
    hx = x_max-x_min
    hy = y_max-y_min
    if(hx > hy):
        y_max += (hx-hy)/2
        y_min -= (hx-hy)/2
    else:
        x_max += (hy-hx)/2
        x_min -= (hy-hx)/2
    x = np.arange(x_min, x_max, h)
    y = np.arange(y_min, y_max, h)
    p0 = fun(x, y, X[0], Y[0])
    p1 = fun(x, y, X[1], Y[1])
    p2 = fun(x, y, X[2], Y[2])
    z0 = p0[np.maximum(p0[:, 0], p1[:, 0], p2[:, 0])==p0[:, 0]]
    z1 = p1[np.maximum(p0[:, 0], p1[:, 0], p2[:, 0])==p1[:, 0]]
    z2 = p2[np.maximum(p0[:, 0], p1[:, 0], p2[:, 0])==p2[:, 0]]
    for i in range(len(z0)):
        for j in range(len(z0)):
            if(z0[i][j]==0):
                z0[i][j] = np.nan
            if(z1[i][j]==0):
                z1[i][j] = np.nan
            if(z2[i][j]==0):
                z2[i][j] = np.nan
            
    x, y = np.meshgrid(x, y)
    y = np.transpose(y)
    x = np.transpose(x)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x, y, z0)
    ax.scatter(x, y, z1)
    ax.scatter(x, y, z2)

def pltHistogram(name, X, Y, h):
    x = X[0]
    y = Y[0]
    for i in range(len(X)-1):
        x = np.concatenate((x, X[i+1]), axis=0)
        y = np.concatenate((y, Y[i+1]), axis=0)
    hist, xedges, yedges = np.histogram2d(x, y, bins=(int((np.max(x)-np.min(x))/h), int((np.max(x)-np.min(x))/h)))
    xpos, ypos = np.meshgrid((xedges[:-1]+xedges[1:])/2, (yedges[:-1]+yedges[1:])/2)
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)
    dx = xedges [1] - xedges [0]
    dy = yedges [1] - yedges [0]
    dz = np.zeros_like(hist.flatten())
    cmap = plt.get_cmap('jet')
    rgba = [cmap(0.1) for k in dz]
    for i in range(len(X)):
        tmp_hist, xedges, yedges = np.histogram2d(X[i], Y[i], bins=(xedges, yedges))
        tmp_hist = tmp_hist.flatten()/(h**2 * len(X[i]))
        for j in range(len(dz)):
            if(tmp_hist[j] > dz[j]):
                dz[j] = tmp_hist[j]
                rgba[j] = cmap(i/len(X))
    fig = plt.figure(name)
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba)

# ----------------------------------------------------------------------------
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

plt.figure(1)
plt.plot(x1, y1, '.')
plt.plot(x2, y2, '.')
plt.plot(x3, y3, '.')

# ----------------------------------------------------------------------------
X = np.concatenate(([x1], [x2], [x3]), axis=0)
Y = np.concatenate(([y1], [y2], [y3]), axis=0)

h1 = 0.09
h2 = 0.3
h3 = 0.6

# pltHistogram("histogram", X, Y, h3)
plting(X, Y, h1, Histogram)
plting(X, Y, h2, Histogram)
plting(X, Y, h3, Histogram)


# ----------------------------------------------------------------------------
# p_KDE1 = KDE(x1, y1, h3, N, D)
# p_KDE2 = KDE(x2, y2, h3, N, D)
# p_KDE3 = KDE(x3, y3, h3, N, D)

# print(len(p_KDE1))

# plt.figure(3)
# plt.hist(p_KDE1, bins=25, alpha=0.7)
# plt.hist(p_KDE2, bins=25, alpha=0.7)
# plt.hist(p_KDE3, bins=25, alpha=0.7)
plt.show()
