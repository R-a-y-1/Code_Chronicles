import numpy as np
import matplotlib.pyplot as plt

# Let's make a bunch of functions!

def summer(obs, exp):        # first to make a summing function
    """
    Does a weird summation on two lists

    obs: array-type, observed value
    exp: array-type, expected value
    note: obs and exp must be of same length
    """
    if type(obs) != list:
        print('This function only accepts lists!')
        return
    
    result = list()

    for y in obs:
        for x in exp:
            test = (y - x)/x
            result.append(test)

    return print(sum(result))

def qikplot(x, y):          # quick plot function
    """
    A function to make a quick and dirty plot of x, and y. \\
    x and y must both be a value or 1-dimensional array.

    x: x-data
    y: y-data 
    """
    plt.plot(x, y, 'o', label = 'Data')
    show = plt.show()
    return show

if __name__ == "__main__":
    summer(list(range(1,100)), list(range(100, 1)))