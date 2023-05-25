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

n = 500


def lipschitzEstimate(L, i, k, x):
    norm = pow(numpy.linalg.norm(df(i, x)), 2)
    if norm > pow(10, -8):
        if f(i, (x - (1 / pow(L, k)) * df(i, x))) <= f(i, x) - 1 / (2 * pow(L, k)) * norm:
            return False
        else:
            return True
    else:
        return False


def stepSize(n):
    return 1 / n


def sag(iter, initalL):
    x = np.zeros(dim)
    d = np.zeros(dim)
    y = []
    round = 1
    for i in range(1, n):
        y.append(numpy.zeros(dim))
    plotdata = []
    L = initalL

    for k in range(1, iter):
        print('\rSAG iteration: ', k, '/', iter, end="")
        i = random.randint(1, n)

        # calculate step size
        if lipschitzEstimate(L, i, k, x):
            L = L * 2

        d = d - y[i] + df(i, x)
        y[i] = df(i, x)
        x = x - ((stepSize(L)) / n) * d
        round = round + 1

        # calculate what we want to minimize:
        curr = 0
        for j in range(1, n):
            curr += f(j, x)
        curr = curr / n
        plotdata.append(curr)
    plt.plot(range(1, iter), plotdata, label="SAG")
    print('\r', iter, 'SAG iterations finished', end="")
    print('\n')
    return x


def sg(iter, initialL):
    x = np.zeros(dim)
    plotdata = []
    round = 1
    L = initialL

    for k in range(1, iter):
        print('\rSG iteration: ', k, '/', iter, end="")
        i = random.randint(1, n)
        # calculate step size
        if lipschitzEstimate(L, i, k, x):
            L = L * 2
        x = x - (stepSize(L)) * df(i, x)
        round = round + 1

        # calculate what we want to minimize:
        curr = 0
        for j in range(1, n):
            curr += f(j, x)
        curr = curr / n
        plotdata.append(curr)
    plt.plot(range(1, iter), plotdata, label="SG")
    print('\r', iter, 'SG iterations finished', end="")
    print('\n')
    return x


def fg(iter, initalL):
    x = np.zeros(dim)
    plotdata = []
    round = 1
    L = initalL

    for k in range(1, iter):
        print('\rFG iteration: ', k, '/', iter, end="")
        updateL = True
        grd = np.zeros(dim)
        for i in range(1, n):
            grd = grd + df(i, x)
            # calculate step size
            if not lipschitzEstimate(L, i, k, x):
                updateL = False
        grd = (1/n)*grd
        if updateL:
            L = L * 2
        x = x - stepSize(L) * grd
        round = round + 1

        # calculate what we want to minimize:
        curr = 0
        for j in range(1, n):
            curr += f(j, x)
        curr = curr / n
        plotdata.append(curr)
    plt.plot(range(1, iter), plotdata, label="FG")
    print('\r', iter, 'FG iterations finished', end="")
    print('\n')
    return x


def f(i, x):
    res = 0
    # calculate the vector product piecewise
    for k in range(1, dim):
        res = res + a[i][k] * x[k]
    res = res - b[i]
    res = res * res
    return res


def df(i, x):
    res = np.zeros(dim)
    # print(a.shape)
    for k in range(1, dim):
        # print('a: ',a.shape)
        # print('x: ',x.shape)
        # print('b: ',b.shape)
        res[k] = 2 * a[i][k] * np.sqrt(f(i, x))
    return res


iters = 100
L = 0.1
# sag(iters, L)
# sg(iters, L)
fg(iters, L)
plt.legend()
name = 'SAG-sim-' + str(iters) + '-' + str(L) + '.png'
plt.savefig(name, dpi=600)
plt.show()
