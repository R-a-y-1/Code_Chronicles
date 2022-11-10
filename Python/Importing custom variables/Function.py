import numpy as np
import matplotlib.pyplot as plt

# Let's make a bunch of functions!

def chisq(obs, exp):        # first to make a chisquare test
    """
    Does a chi squared test on two variables

    obs: float or array-type, observed value
    exp: float or array-type, expected value
    note: obs and exp must be of same length
    """
    result = list()

    for obs in obs:
        for exp in exp:
            test = (obs - exp)/exp
            result.append(test)
    return sum(result)

def qikplot(x, y):          # quick plot function
    """
    A function to make a quick and dirty plot of x, and y. \\
    x and y must both be a value or 1-dimensional array.

    x: x-data
    y: y-data 
    """
    graph = plt.plot(x, y, 'o', label = 'Data')
    return graph

