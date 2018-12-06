import math
import numpy as np

### Note ###

# yHat is prediction
# y is the target (true label)


### Functions ###

def CrossEntropy(yHat, y):
    eps = 0.000001 # maximum loss is -log(eps)
    loss = 0
    if y == 1:
      # y is positive class
      if (yHat == 0):
        yHat = eps
      loss = -log(yHat)
    else:
      # y is negative class
      if (yHat = 1):
        yHat = 1 - eps
      loss = -log(1 - yHat)
    # ensure that loss is monotonically increasing
    return min(-log(eps), loss)


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
