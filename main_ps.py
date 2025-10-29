import numpy as np
import matplotlib.pyplot as plt
from Perceptron import Perceptron


data = np.loadtxt("../spiral_d.csv", delimiter=',')
controle_figura = True
# print(data)


X = data[:, :-1]
y = data[:, -1:]
N, p = X.shape

COLORS = ['#f6cfff', '#b8e6fe', '#c4b4ff', '#fccee8', '#96f7e4', '#a4f4cf', '#fef3c6', '#ffccd3']

# fig = plt.figure(1)
# ax = fig.add_subplot(projection='3d')
# ax.scatter(data[:,0],data[:,1],data[:,2], edgecolor='k')
# ax.set_title("Todo o conjunto de dados.")
# ax.set_xlabel("Variavel 1")
# ax.set_ylabel("Variavel 2")
# ax.set_zlabel("Resultado")

# plt.show()

# X = np.array([
# [1, 1],
# [0, 1],
# [0, 2],
# [1, 0],
# [2, 2],
# [4, 1.5],
# [1.5, 6],
# [3, 5],
# [3, 3],
# [6, 4]])

# y = np.array([
# [1],
# [1],
# [1],
# [1],
# [1],
# [-1],
# [-1],
# [-1],
# [-1],
# [-1]])

ps = Perceptron(X.T, y)
ps.fit()
plt.show()