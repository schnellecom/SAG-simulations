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


# n = 5000


def sag(iter, alpha):
    x = np.zeros(dim)
    d = np.zeros(dim)
    y = []
    round = 1
    for i in range(1, n):
        y.append(numpy.zeros(dim))
    plotdata = []

    for k in range(1, iter):
        print('\rSAG iteration: ', k, '/', iter, end="")
        i = random.randint(1, n)
        d = d - y[i] + df(i, x)
        y[i] = df(i, x)
        x = x - ((1 / round) / n) * d
        round = round + 1

        # calculate what we want to minimize:
        curr = 0
        for j in range(1, n):
            curr += f(j, x)
        curr = curr / n
        plotdata.append(curr)
    plt.plot(range(1, iter), plotdata, label="SAG")
    print('\r', iter, 'SAG iterations finished/', end="")
    print('\n')
    return x


def sg(iter, alpha):
    x = np.zeros(dim)
    plotdata = []
    round = 1

    for k in range(1, iter):
        print('\rSG iteration: ', k, '/', iter, end="")
        i = random.randint(1, n)
        x = x - (1 / round) * df(i, x)
        round = round + 1

        # calculate what we want to minimize:
        curr = 0
        for j in range(1, n):
            curr += f(j, x)
        curr = curr / n
        plotdata.append(curr)
    plt.plot(range(1, iter), plotdata, label="SG")
    print('\r', iter, 'SG iterations finished/', end="")
    print('\n')
    return x


def fg(iter, alpha):
    x = np.zeros(dim)
    plotdata = []
    round = 1

    for k in range(1, iter):
        print('\rFG iteration: ', k, '/', iter, end="")
        grd = 0.
        for i in range(1, n):
            grd = grd + df(i, x)
        x = x - (1 / round) / n * grd
        round = round + 1

        # calculate what we want to minimize:
        curr = 0
        for j in range(1, n):
            curr += f(j, x)
        curr = curr / n
        plotdata.append(curr)
    plt.plot(range(1, iter), plotdata, label="FG")
    print('\r', iter, 'FG iterations finished/', end="")
    print('\n')
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
        res[k] = 2 * a[i][1] * (a[i][k] * x[k] - b[i])
    return res


iters = 100
alpha = 1 / 10
sag(iters, alpha)
sg(iters, alpha)
fg(iters, alpha)
plt.legend()
name = 'SAG-sim-' + str(iters) + '-' + str(alpha) + '.png'
plt.savefig(name, dpi=600)
plt.show()
