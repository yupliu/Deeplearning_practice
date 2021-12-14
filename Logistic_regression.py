import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()
x = iris.data[:,(2,3)]
y = (iris.target == 0).astype(np.int32)
pc = Perceptron()
pc.fit(x,y)
y_pred = pc.predict([[2,0.5]])
print(y_pred)