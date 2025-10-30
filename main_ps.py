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

print(f"Iniciando simulação de Monte Carlo com {R} rodadas...")
for r in range(R):
    
    idx = np.random.permutation(N)
    Xr_ = X[idx,:]
    yr = y[idx, :]
    

    # Particionamento do conjunto de dados (80% treino, 20% teste)
    split_idx = int(N * 0.8)
    X_treino = Xr[:int(N*.8), :]
    y_treino = yr[:int(N*.8), :]
    
    X_teste = Xr[int(N*.8):, :] 
    y_teste = yr[int(N*.8):, :]
    
    ps = Perceptron(X_treino.T, y_treino)
    ps = ADALINE(X_treino.T, y_treino)

ps = Perceptron(X.T, y)
# ps = ADALINE(X.T, y)
ps.fit()
plt.show()