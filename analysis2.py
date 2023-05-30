import random
import sys

import numpy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import double

# get the data, i.e. the a_i from the paper
data = pd.read_csv('data/phy_train.dat', sep='\t', header=None, usecols=[i for i in range(2, 80)])
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
n = 500


def lipschitzCheck(x, L, i):
    """
    checks if the lipschitzconstant which is currently guessed should be doubled
    c.f. 4.6 in the paper
    :param x: the current x^k
    :param L: the current l^k
    :param i: the sample considered
    :returns bool: true if should be doubled, false otherwise
    """


def g(x):
    """
    evaluates the function g at vector x
    c.f. section 1, (1)
    :param x: the vector to evaluate at
    :returns float: g evaluated at x
    """


def f(i, x):
    """
    evaluates the function f including a regularizer at sample i
    c.f. section 1 (1)
    :param i: the sample considered
    :param x:
    :returns float:
    """


def dg(x):
    """
    the gradient of g at vector x
    :param x: the vector where the gradient should be calculated at
    :returns: np.array: the gradient of g at x
    """


def df(i, x):
    """
    the gradient of f_i at vector x so f(i,x)
    :param x: the vector where the gradient should be calculated at
    :returns: np.array: the gradient of f at x
    """


def fullGradient(iters, initialX, initialL):
    """
    perform the full gradient descent of the function g
    c.f. section 1 (3)
    :param iters: number of iterations to perform
    :param initialX: initial guess of x
    :param initialL: initial guess of L
    """
