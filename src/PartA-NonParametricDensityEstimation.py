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
    z, x, y = np.histogram2d(X, Y, bins=(x, y))
    return z/((x[1]-x[0])*len(X))

def Parzen_windows(x, y, X, Y):
    h = x[1]-x[0]
    p = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            for n in range(len(X)):
                p[i][j] += K([(x[i]-X[n])/h, (y[j]-Y[n])/h])
    p = p/(len(X)*h**2)
    return p

sigma = 0
def Gaussian_kernel(x, y, X, Y):
    x, y = np.meshgrid(x, y)
    y = np.transpose(y)
    x = np.transpose(x)
    muu = 0.0
    dst = np.sqrt((x-X[0])**2+(y-Y[0])**2)
    gauss = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )
    p = gauss
    for i in range(1, len(X)):
        dst = np.sqrt((x-X[i])**2+(y-Y[i])**2)
        gauss = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )
        p += gauss
    return p

def _Gaussian_kernel(x, y, X, Y):
    muu = 0.0
    dst = np.sqrt((x-X[0])**2+(y-Y[0])**2)
    gauss = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )
    p = gauss
    for i in range(1, len(X)):
        dst = np.sqrt((x-X[i])**2+(y-Y[i])**2)
        gauss = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )
        p += gauss
    return p

fork = 0
def KNN(x, y, X, Y):
    p = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            dist = np.sqrt((X-x[i])**2+(Y-y[j])**2)
            R = np.partition(dist, fork-1)[fork-1]
            p[i][j] = fork/(len(X)*np.pi*R**2)
    return p

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
    p0 = fun(np.copy(x), np.copy(y), X[0], Y[0])
    p1 = fun(np.copy(x), np.copy(y), X[1], Y[1])
    p2 = fun(np.copy(x), np.copy(y), X[2], Y[2])
    
    p0[np.maximum(np.maximum(p0, p1), p2)!=p0]=0
    p1[np.maximum(np.maximum(p0, p1), p2)!=p1]=0
    p2[np.maximum(np.maximum(p0, p1), p2)!=p2]=0
    p0[p0==0] = np.nan
    p1[p1==0] = np.nan
    p2[p2==0] = np.nan
            
    x, y = np.meshgrid(x, y)
    y = np.transpose(y)
    x = np.transpose(x)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x, y, p0)
    ax.scatter(x, y, p1)
    ax.scatter(x, y, p2)

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

# plt.figure()
# plt.plot(x1, y1, '.', label='class_1')
# plt.plot(x2, y2, '.', label='class_2')
# plt.plot(x3, y3, '.', label='class_3')
# plt.title('Datasets')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()

# ----------------------------------------------------------------------------
X = np.concatenate(([x1], [x2], [x3]), axis=0)
Y = np.concatenate(([y1], [y2], [y3]), axis=0)

h1 = 0.09
h2 = 0.3
h3 = 0.6

# plting(X, Y, h1, Histogram)
# plting(X, Y, h2, Histogram)
# plting(X, Y, h3, Histogram)

# plting(X, Y, h1, Parzen_windows)
# plting(X, Y, h2, Parzen_windows)
# plting(X, Y, h3, Parzen_windows)

# sigma = 0.2
# plting(X, Y, h3, Gaussian_kernel)
# sigma = 0.6
# plting(X, Y, h3, Gaussian_kernel)
# sigma = 0.9
# plting(X, Y, h3, Gaussian_kernel)

# fork = 1
# plting(X, Y, h3, KNN)
# fork = 9
# plting(X, Y, h3, KNN)
# fork = 99
# plting(X, Y, h3, KNN)

# best value for h:
# sigma = 0.6
# best_h = 1.06*sigma*N**(-1.0/5)
# plting(X, Y, h3, Gaussian_kernel)

train_x1 = x1[:int(N*0.9)]
train_y1 = y1[:int(N*0.9)]
train_x2 = x2[:int(N*0.9)]
train_y2 = y2[:int(N*0.9)]
train_x3 = x3[:int(N*0.9)]
train_y3 = y3[:int(N*0.9)]
test_x1 = x1[int(N*0.9):]
test_y1 = y1[int(N*0.9):]
test_x2 = x2[int(N*0.9):]
test_y2 = y2[int(N*0.9):]
test_x3 = x3[int(N*0.9):]
test_y3 = y3[int(N*0.9):]

sigma = 0.6
p1_1 = _Gaussian_kernel(test_x1, test_y1, train_x1, train_y1)
p1_2 = _Gaussian_kernel(test_x1, test_y1, train_x2, train_y2)
p1_3 = _Gaussian_kernel(test_x1, test_y1, train_x3, train_y3)
T1 = len(test_x1[p1_1 >= np.maximum(p1_2, p1_3)])

p2_1 = _Gaussian_kernel(test_x2, test_y2, train_x1, train_y1)
p2_2 = _Gaussian_kernel(test_x2, test_y2, train_x2, train_y2)
p2_3 = _Gaussian_kernel(test_x2, test_y2, train_x3, train_y3)
T2 = len(test_x2[p2_2 >= np.maximum(p2_1, p2_3)])

p3_1 = _Gaussian_kernel(test_x3, test_y3, train_x1, train_y1)
p3_2 = _Gaussian_kernel(test_x3, test_y3, train_x2, train_y2)
p3_3 = _Gaussian_kernel(test_x3, test_y3, train_x3, train_y3)
T3 = len(test_x3[p3_3 >= np.maximum(p3_1, p3_2)])

print("accuracies:")
print((T1+T2+T3)/150)

# plt.figure()
# plt.plot(train_x1, train_y1, '.', label='class_1')
# plt.plot(train_x2, train_y2, '.', label='class_2')
# plt.plot(train_x3, train_y3, '.', label='class_3')
# plt.title('Train_datasets')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.figure()
# plt.plot(test_x1, test_y1, '.', label='class_1')
# plt.plot(test_x2, test_y2, '.', label='class_2')
# plt.plot(test_x3, test_y3, '.', label='class_3')
# plt.title('Test_datasets')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()

# plt.figure()
# plt.plot(test_x1[p1_1 >= np.maximum(p1_2, p1_3)], test_y1[p1_1 >= np.maximum(p1_2, p1_3)], '.', label='Test_datasets class_1 (T)')
# plt.plot(test_x2[p2_2 >= np.maximum(p2_1, p2_3)], test_y2[p2_2 >= np.maximum(p2_1, p2_3)], '.', label='Test_datasets class_2 (T)')
# plt.plot(test_x3[p3_3 >= np.maximum(p3_1, p3_2)], test_y3[p3_3 >= np.maximum(p3_1, p3_2)], '.', label='Test_datasets class_3 (T)')
# plt.plot(test_x1[p1_1 <np.maximum(p1_2, p1_3)], test_y1[p1_1 <np.maximum(p1_2, p1_3)], 'o', label='Test_datasets class_1 (F)')
# plt.plot(test_x2[p2_2 <np.maximum(p2_1, p2_3)], test_y2[p2_2 <np.maximum(p2_1, p2_3)], 'o', label='Test_datasets class_2 (F)')
# plt.plot(test_x3[p3_3 <np.maximum(p3_1, p3_2)], test_y3[p3_3 <np.maximum(p3_1, p3_2)], 'o', label='Test_datasets class_3 (F)')
# plt.title('Gaussian kernel result')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()

plt.show()
