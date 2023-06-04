import random
import sys

import numpy
import numpy as np
import pandas as pd
import sympy
from matplotlib import pyplot as plt
from numpy import double
from sympy import *

# get the data, i.e. the a_i from the paper
unavailable = [21,22,23,30,45,46,47,56]
# unavailable = []
data = pd.read_csv('data/phy_train.dat', sep='\t', header=None, usecols=[i for i in range(2, 80) if i not in unavailable])
a = data.to_numpy()

# get the targets, i.e. the b_i from the paper
target = pd.read_csv('data/phy_train.dat', sep='\t', header=None, usecols=[1])
b = target.to_numpy()

# get the sizes of the data
# n is number of samples
# dim is dimension of any sample
dim = a.shape[1]
n = b.shape[0]

# for debugging only consider part of the samples
n = 5000

def lipschitzCheck(x, L, i):
    """
    checks if the lipschitzconstant which is currently guessed should be doubled
    c.f. 4.6 in the paper
    :param x: the current x^k
    :param L: the current l^k
    :param i: the sample considered
    :param lam: the lambda of the regularizer
    :returns bool: true if should be doubled, false otherwise
    """


def g(x, lam):
    """
    evaluates the function g at vector x
    c.f. section 1, (1)
    :param x: the vector to evaluate at
    :param lam: the lambda of the regularizer
    :returns float: g evaluated at x
    """
    return (1 / n) * sum([f(i, x, lam) for i in range(n)])


def f(i, x, lam):
    """
    evaluates the function f including a regularizer at sample i
    c.f. section 1 (1)
    :param i: the sample considered
    :param x: the vector to be evaluated
    :param lam: the lambda of the regularizer
    :returns float: the function f_i evaluated at x
    """
    return (np.dot(a[i], x) - b[i]) ** 2 + (lam/2)*np.linalg.norm(x)**2


def dg(x, lam):
    """
    the gradient of g at vector x
    :param x: the vector where the gradient should be calculated at
    :param lam: the lambda of the regularizer
    :returns: np.array: the gradient of g at x
    """
    return np.array([(1/n)*np.sum([(2*(np.dot(a[i], x) - b[i])*a[i][k]) for i in range(n)])+(lam/2)*x[k] for k in range(dim)])


def df(i, x, lam):
    """
    the gradient of f_i at vector x so f(i,x)
    :param i: the sample to consider
    :param x: the vector where the gradient should be calculated at
    :param lam: the lambda of the regularizer
    :returns: np.array: the gradient of f at x
    """
    const = 2 * (np.dot(a[i], x) - b[i])
    return [(a[i][k] * const + (lam/2)*x[k]) for k in range(dim)]


def fullGradient(iters, initialX, initialL, lam):
    """
    perform the full gradient descent of the function g
    c.f. section 1 (3)
    :param iters: number of iterations to perform
    :param initialX: initial guess of x
    :param initialL: initial guess of L
    :param lam: the lambda of the regularizer
    """

    L = initialL
    plotdata = []

    xk = initialX
    for k in range(iters):
        xk = xk - L*dg(xk, lam)
        plotdata.append(g(xk, lam))
        print('\rFG iteration: ', k, '/', iters, end="")

    plt.plot(range(1, iters + 1), plotdata, label="FG")
    print('\r', iters, 'FG iterations finished', end="")
    print('\n')

lam = 2
iters = 10
x = np.ones(dim)
L = 0.00001

fullGradient(iters, x, L, lam)

plt.legend()
name = 'SAG-sim2-' + str(iters) + '-' + str(L) + '.png'
plt.savefig(name, dpi=600)
plt.show()
