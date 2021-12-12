import numpy as np

def sigmoid(x):
    tmp = 1/(1+np.exp(-x))
    return tmp

x = np.array([1,2,3])
print(sigmoid(x))

def sigmoid_derivative(x):
    ds = sigmoid(x)*(1-sigmoid(x))
    return ds

print(sigmoid_derivative(x))

print(x.shape)

def image2vect(x):
    nx = x.shape[0]
    ny = x.shape[1]
    nz = x.shape[2]
    return (x.reshape(nx*ny*nz,1))


image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])


print(image2vect(image))

