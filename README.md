# Non-Parametric Density Estimation, and PCA

## Setup and run:

* Install python3

* Install python library:
```bash
$ pip install pandas
$ pip install numpy
$ pip install matplotlib
```

* Clone and run:
```bash
$ git clone https://github.com/VakhshooriEhsan/SPR-Fall2020-HW4-Non-Parametric-Density-Estimation-and-PCA.git
$ cd SPR-Fall2020-HW4-Non-Parametric-Density-Estimation-and-PCA/src
$ python PartA-NonParametricDensityEstimation.py
```

## A. Non-Parametric Density Estimation

### 0. Generate dataset
![plot](https://github.com/VakhshooriEhsan/SPR-Fall2020-HW2-Logistic-SoftmaxRegression-MC--BayesianClassification/blob/master/docs/imgs/Dataset.PNG?raw=true)

### 1. Implement PDF estimation using h = 0.09, 0.3, 0.6

* 1.1. Histogram
![plot](https://github.com/VakhshooriEhsan/SPR-Fall2020-HW2-Logistic-SoftmaxRegression-MC--BayesianClassification/blob/master/docs/imgs/Histogram.PNG?raw=true)

* 1.2. Parzen Window
![plot](https://github.com/VakhshooriEhsan/SPR-Fall2020-HW2-Logistic-SoftmaxRegression-MC--BayesianClassification/blob/master/docs/imgs/Parzen_windows.PNG?raw=true)

* 1.2.1 Gaussian kernel (Standard Deviations of 0.2)
![plot](https://github.com/VakhshooriEhsan/SPR-Fall2020-HW2-Logistic-SoftmaxRegression-MC--BayesianClassification/blob/master/docs/imgs/Gaussian_kernel_1.PNG?raw=true)

* 1.2.2 Gaussian kernel (Standard Deviations of 0.6)
![plot](https://github.com/VakhshooriEhsan/SPR-Fall2020-HW2-Logistic-SoftmaxRegression-MC--BayesianClassification/blob/master/docs/imgs/Gaussian_kernel_2.PNG?raw=true)

* 1.2.3 Gaussian kernel (Standard Deviations of 0.9)
![plot](https://github.com/VakhshooriEhsan/SPR-Fall2020-HW2-Logistic-SoftmaxRegression-MC--BayesianClassification/blob/master/docs/imgs/Gaussian_kernel_3.PNG?raw=true)

* 1.3.1 KNN (Fork = 1)
![plot](https://github.com/VakhshooriEhsan/SPR-Fall2020-HW2-Logistic-SoftmaxRegression-MC--BayesianClassification/blob/master/docs/imgs/KNN_1.PNG?raw=true)

* 1.3.2 KNN (Fork = 9)
![plot](https://github.com/VakhshooriEhsan/SPR-Fall2020-HW2-Logistic-SoftmaxRegression-MC--BayesianClassification/blob/master/docs/imgs/KNN_2.PNG?raw=true)

* 1.3.3 KNN (Fork = 99)
![plot](https://github.com/VakhshooriEhsan/SPR-Fall2020-HW2-Logistic-SoftmaxRegression-MC--BayesianClassification/blob/master/docs/imgs/KNN_3.PNG?raw=true)

### 2. Find the best value for h

![plot](https://github.com/VakhshooriEhsan/SPR-Fall2020-HW2-Logistic-SoftmaxRegression-MC--BayesianClassification/blob/master/docs/imgs/Best-value-for-h.PNG?raw=true)

```
Best value for h:
0.18351142803139756
```

### 3. Gaussian kernel estimation

![plot](https://github.com/VakhshooriEhsan/SPR-Fall2020-HW2-Logistic-SoftmaxRegression-MC--BayesianClassification/blob/master/docs/imgs/Train-Test-Total_accuracies.PNG?raw=true)

```
accuracies:
0.9333333333333333
```
