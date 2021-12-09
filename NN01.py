import numpy as np

def sigmoid(x):
    tmp = 1/(1+np.exp(-x))
    return tmp

x = np.array([1,2,3])
print(sigmoid(x))