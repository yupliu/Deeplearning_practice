from cgi import test
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

from sklearn.datasets import load_boston
data_x,data_y = load_boston(return_X_y=True)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(data_x,data_y,test_size=0.2,random_state=42)

