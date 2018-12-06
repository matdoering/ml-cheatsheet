import math
import numpy as np

### Note ###

# yHat is prediction
# y is the target (true label)


### Functions ###

def CrossEntropy(yHat, y):
    if yHat == 1:
      return 0
    else:
      return -log(1 - yHat)


def Dice(yHat, y):
    total = np.sum(y, dim=1) + np.sum(yHat, dim=1)
    intersection = np.sum(y * yHat, dim=1)
    dice = (2.0 * intersection) / (total + 1e-7)
    return np.mean(dice)


def Hinge(yHat, y):
    return np.max(0, 1 - yHat * y)


def Huber(yHat, y, delta=1.):
    return np.where(np.abs(y-yHat) < delta,.5*(y-yHat)**2 , delta*(np.abs(y-yHat)-0.5*delta))


def KLDivergence(yHat, y):
    pass


def L1(yHat, y):
    return np.sum(np.absolute(yHat - y))


def L2(yHat, y):
    return np.sum((yHat - y)**2)


def MLE(yHat, y):
    pass


def MSE(yHat, y):
    return np.sum((yHat - y)**2) / y.size


### Derivatives ###

def MSE_prime(yHat, y):
    return yHat - y
