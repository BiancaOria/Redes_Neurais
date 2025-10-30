import numpy as np
import matplotlib.pyplot as plt
from Perceptron import Perceptron
from Adaline import ADALINE



data = np.loadtxt("../spiral_d.csv", delimiter=',')
controle_figura = True
# print(data)


X = data[:, :-1]
y = data[:, -1:]
N, p = X.shape


ps = ADALINE(X.T, y)
ps.fit()
plt.show()