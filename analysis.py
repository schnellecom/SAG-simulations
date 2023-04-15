import random
import sys

import numpy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('data/phy_train.dat', sep='\t', header=None, usecols=[i for i in range(2, 80)])
# data = data.transpose()
a = data.to_numpy()

target = pd.read_csv('data/phy_train.dat', sep='\t', header=None, usecols=[1])
# target = target.transpose()
b = target.to_numpy()

# print(data)
# print('target: ',target.shape)
# print(target)
dim = a.shape[1]
n = b.shape[0]


def sag(iter, alpha):
    x = np.zeros(dim)
    d = np.zeros(dim)
    y = []
    for i in range(1, n):
        y.append(numpy.zeros(dim))
    plotdata = []

    for k in range(1, iter):
        print('\rSAG iteration: ', k, '/', iter, end="")
        i = random.randint(1, n)
        d = d - y[i] + df(i, x)
        y[i] = df(i, x)
        x = x - (alpha / n) * d

        # calculate what we want to minimize:
        curr = 0
        for j in range(1, n):
            curr += f(j, x)
        curr = curr/n
        plotdata.append(curr)
    plt.plot(range(1, iter), plotdata, label="SAG")
    return x


def sg(iter, alpha):
    x = np.zeros(dim)
    plotdata = []

    for k in range(1, iter):
        print('\rSG iteration: ', k, '/', iter, end="")
        i = random.randint(1, n)
        x = x - alpha*df(i, x)

        # calculate what we want to minimize:
        curr = 0
        for j in range(1, n):
            curr += f(j, x)
        curr = curr/n
        plotdata.append(curr)
    plt.plot(range(1, iter), plotdata, label="SG")
    return x


def f(i, x):
    res = 0
    # calculate the vector product piecewise
    for k in range(1, dim):
        res = a[i][k] * x[k]
    res -= b[i]
    res = res * res
    return res


def df(i, x):
    res = np.zeros(dim)
    # print(a.shape)
    for k in range(1, dim):
        # print('a: ',a.shape)
        # print('x: ',x.shape)
        # print('b: ',b.shape)
        res[k] = 2 * (a[i][k] * x[k] - b[k]) * a[i][k]
    return res


sag(20, 10)
# sg(20, 1/2)
plt.legend()
plt.show()