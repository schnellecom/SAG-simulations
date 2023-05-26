import random
import sys

import numpy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import double

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
        i = random.randint(0, n-1)

        # calculate step size
        if lipschitzEstimate(L, i, k, x):
            L = L * 2

        d = d - y[i] + df(i, x)
        y[i] = df(i, x)
        x = x - stepSize(L) * d
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
    L = initialL

    for k in range(1, iter):
        print('\rSG iteration: ', k, '/', iter, end="")
        i = random.randint(1, n)
        # calculate step size
        if lipschitzEstimate(L, i, k, x):
            L = L * 2
        x = x - 1/k * df(i, x)

        # calculate what we want to minimize:
        curr = 0
        for j in range(1, n):
            curr += f(j, x)
        curr = curr / n
        plotdata.append(curr)
    plt.semilogy(range(1, iter), plotdata, label="SG")
    print('\r', iter, 'SG iterations finished', end="")
    print('\n')
    return x


def fg(nIter, initalL):
    x = np.ones(dim)
    plotdata = []
    lipConst = initalL

    for k in range(1, nIter + 1):
        print('\rFG iteration: ', k, '/', nIter, end="")
        updateL = True
        grd = np.zeros(dim)
        for i in range(n):
            grd += df(i, x)
            # calculate step size
            # if not lipschitzEstimate(lipConst, i, k, x):
            #     updateL = False
        grd = (1/n)*grd
        # if updateL:
        #     lipConst = lipConst * pow(2, (-1 / n))
        x = x - 1/k * grd

        # calculate what we want to minimize:
        curr = 0
        for j in range(n):
            curr += f(j, x)
        curr = curr / n
        plotdata.append(curr)
    plt.semilogy(range(1, nIter+1), plotdata, label="FG")
    print('\r', nIter, 'FG iterations finished', end="")
    print('\n')
    return x


def f(i, x):
    res = 0
    # calculate the vector product piecewise
    for k in range(dim):
        res += a[i][k] * x[k]
    res -= b[i]
    return res * res


def df(i, x):
    res = np.zeros(dim)
    # print(a.shape)
    const = 0
    for t in range(dim):
        const += a[i][t] * x[t]
    const -= b[i]
    for k in range(dim):
        # print('a: ',a.shape)
        # print('x: ',x.shape)
        # print('b: ',b.shape)
        res[k] = 2 * a[i][k] * const
    return res


iters = 3
L = 0.1
# sag(iters, L)
#sg(iters, L)
fg(iters, L)
plt.legend()
name = 'SAG-sim-' + str(iters) + '-' + str(L) + '.png'
plt.savefig(name, dpi=600)
plt.show()
