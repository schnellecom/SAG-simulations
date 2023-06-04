import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

file = 'data/phy_train.dat'

## SAMPLES
samples = pd.read_csv(file, sep='\t', header=None, usecols=[*range(2, 80)])
# samples = samples.transpose()
samples = samples.to_numpy()
samples_dim = samples.shape[1]
n_samples = samples.shape[0]

n_samples = 5000

## LABELS
targets = pd.read_csv(file, sep='\t', header=None, usecols=[1])
targets = targets.to_numpy()
# targets = targets.transpose()
targets = np.fromiter((1 if t > 0 else -1 for t in targets), dtype=float)

# data = zip(samples,targets)
# print("data:", *data)

# for regularizer
lam = 0.5


# loss function f_i for sample i
def f_i(i, x):
    squared_residual = (np.dot(samples[i], x) - targets[i]) ** 2
    regularizer = (lam / 2) * np.linalg.norm(x) ** 2
    loss = squared_residual + regularizer
    return loss


# gradient of loss function f_i for sample i
def dfdx_i(i, x):
    d_squared_residual_dx = 2.0 * (np.dot(samples[i], x) - targets[i]) * samples[i]
    d_regularizer_dx = lam * x
    d_loss_dx = d_squared_residual_dx + d_regularizer_dx
    return d_loss_dx


# average of function g(i,x) over samples i
def average(g, x):
    return (1.0 / n_samples) * np.sum(np.fromiter((g(i, x) for i in range(n_samples)), np.ndarray))


# loss function f -- average of the f_i
def f(x):
    return average(f_i, x)


# gradient of loss function f -- average of the gradients of the f_i
def dfdx(x):
    return average(dfdx_i, x)


def full_gradient_descent(n_iter, initial_lipschitz_constant):
    L = initial_lipschitz_constant

    # keep track of iterands
    estimates = []

    # initial estimate
    x = np.ones(samples_dim)
    estimates.append(f(x))

    # gradient descent loop
    for k in range(n_iter):
        # print(f'iteration:\t{k+1} / {n_iter}')

        learning_rate = 1.0 / (L)
        gradient = dfdx(x)
        print(f'\rfull gradient:\t{np.linalg.norm(gradient)}\t{k+1}', end="")
        x -= learning_rate * gradient
        y = f(x)
        # print(f'value:\t\t{np.linalg.norm(y)}')

        estimates.append(y)

    return estimates


def stochastic_averaging_gradient_descent(n_iter, initial_lipschitz_constant):
    L = initial_lipschitz_constant

    # keep track of iterands
    estimates = []

    # initial estimate
    x = np.ones(samples_dim)
    estimates.append(f(x))

    gradient_sum = np.zeros(samples_dim)
    gradient_updates = [np.zeros(samples_dim) for _ in range(n_samples)]

    # gradient descent loop
    for k in range(n_iter):
        learning_rate = 1.0 / (L)

        # point to be evaluated
        i = random.randint(0, n_samples)

        gradient_sum -= gradient_updates[i]
        gradient_updates[i] = dfdx_i(i, x)
        gradient_sum += gradient_updates[i]

        gradient_average = (1 / n_samples) * gradient_sum

        print(f'\rstochastic averaging gradient:\t{np.linalg.norm(gradient_average)}\t{k+1}', end="")
        x -= learning_rate * gradient_average
        y = f(x)

        estimates.append(y)

    return estimates

def stochastic_averaging_gradient_descent_fancy_initialization(n_iter, initial_lipschitz_constant):
    L = initial_lipschitz_constant

    # keep track of iterands
    estimates = []

    # initial estimate
    x = np.ones(samples_dim)
    estimates.append(f(x))

    dy = dfdx(x)
    gradient_updates = [(dfdx_i(j, x) - dy) for j in range(n_samples)]
    gradient_sum = np.sum(gradient_updates)

    # gradient descent loop
    for k in range(n_iter):
        learning_rate = 1.0 / (L)

        # point to be evaluated
        i = random.randint(0, n_samples)

        gradient_sum -= gradient_updates[i]
        gradient_updates[i] = dfdx_i(i, x)
        gradient_sum += gradient_updates[i]

        gradient_average = (1 / n_samples) * gradient_sum

        print(f'\rstochastic averaging gradient fancy:\t{np.linalg.norm(gradient_average)}\t{k+1}', end="")
        x -= learning_rate * gradient_average
        y = f(x)

        estimates.append(y)

    return estimates


def stochastic_gradient_descent(n_iter, initial_lipschitz_constant):
    L = initial_lipschitz_constant

    # keep track of iterands
    estimates = []

    # initial estimate
    x = np.ones(samples_dim)
    estimates.append(f(x))

    # gradient descent loop
    for k in range(n_iter):
        learning_rate = 1.0 / (L)

        # point to be evaluated
        i = random.randint(0, n_samples)

        gradient = dfdx_i(i, x)
        print(f'\rstochastic gradient:\t{np.linalg.norm(gradient)}\t{k+1}', end="")
        x -= learning_rate * gradient
        y = f(x)

        estimates.append(y)

    return estimates


N_ITERS = 20
LIPSCHITZ = 1e12

plt.figure()

fg_estimates = full_gradient_descent(N_ITERS, LIPSCHITZ)
plt.plot(range(0, N_ITERS + 1), fg_estimates, label="FG")

for run in range(1):
    sg_estimates = stochastic_gradient_descent(N_ITERS, LIPSCHITZ)
    plt.plot(range(0, N_ITERS + 1), sg_estimates, label=f'SG-{run}')

    sg_estimates = stochastic_averaging_gradient_descent(N_ITERS, LIPSCHITZ)
    plt.plot(range(0, N_ITERS + 1), sg_estimates, label=f'SAG-{run}')

    sg_estimates = stochastic_averaging_gradient_descent_fancy_initialization(N_ITERS, LIPSCHITZ)
    plt.plot(range(0, N_ITERS + 1), sg_estimates, label=f'SAGf-{run}')

plt.legend()
name = 'SAG-sim4-' + str(N_ITERS) + '-' + str(LIPSCHITZ) + '.png'
plt.savefig(name, dpi=600)
plt.show()
