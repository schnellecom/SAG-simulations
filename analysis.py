import random

import numpy
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

data = pd.read_csv('data/phy_train.dat', sep='\t', header=None, usecols=[i for i in range(2, 80)])
data = data.transpose()

target = pd.read_csv('data/phy_train.dat', sep='\t', header=None, usecols=[1])
target = target.transpose()

# print(data)
# print(target)
dim = target.size
n = data.shape[1]


def sag(iter):
    x = 0
    d = 0
    y = []
    for i in range(1, n):
        y.append(numpy.zeros(dim))
    plotdata = []

    for k in range(1, iter):
        i = random.randint(1, n)

        plotdata.append(i)
    plt.plot(range(1, iter), plotdata)
    plt.show()
    return x


def f(i, x):
    res = 0
    for j in range(1, i):
        res += x[j] * data[i][j]
    res -= target[i]
    res *= res
    return res


sag(20)
