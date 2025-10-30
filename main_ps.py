import numpy as np
import matplotlib.pyplot as plt
from Perceptron import Perceptron
from Adaline import ADALINE
from Avaliador import Avaliador



data = np.loadtxt("../spiral_d.csv", delimiter=',')


X = data[:, :-1]
y = data[:, -1:]
N, p = X.shape
d = np.array(y).flatten()
XT=X.T

#plot incial
plt.ion()
fig = plt.figure(1)
ax = fig.add_subplot(projection='3d')
ax.scatter(XT[0, (d==1)],
                XT[1, (d==1)],
                d[(d==1)],
                c='r', marker='s', s=120, edgecolor='k',label='Classe 1')

ax.scatter(XT[0, (d==-1)],
                XT[1, (d==-1)],
                d[(d==-1)], 
                c='b', marker='o', s=120, edgecolor='k',label='Classe -1')
margin = 1  # margem extra
x_min, x_max = XT[0].min() - margin, XT[0].max() + margin
y_min, y_max = XT[1].min() - margin, XT[1].max() + margin
z_min, z_max = d.min() - margin, d.max() + margin

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)

ax.set_xlabel("Variável 1")
ax.set_ylabel("Variável 2")
ax.set_zlabel("Resultado")
ax.set_title("Visualização Inicial dos Dados (Espiral 3D)")
ax.legend()

#monte carlo

metricas_acuracia = []
metricas_sensibilidade = []
metricas_especificidade = []
metricas_precisao = []
metricas_f1_score = []

R = 5 #ajustar

print(f"Iniciando simulação de Monte Carlo com {R} rodadas...")
for r in range(R):
    
    idx = np.random.permutation(N)
    Xr = X[idx,:]
    yr = y[idx, :]
    

    # Particionamento do conjunto de dados (80% treino, 20% teste)
    split_idx = int(N * 0.8)
    X_treino = Xr[:split_idx, :]
    y_treino = yr[:split_idx, :]
    
    X_teste = Xr[split_idx:, :] 
    y_teste = yr[split_idx:, :]
    
    
    ps = Perceptron(X_treino.T, y_treino, plot=True, max_epoch=200, learning_rate=0.01)
    ps.fit()
    
    y_pred = ps.predict(X_teste.T)
    
    acc, sens, spec, prec, f1 = Avaliador.calcular_metricas(y_teste, y_pred)
    
    metricas_acuracia.append(acc)
    metricas_sensibilidade.append(sens)
    metricas_especificidade.append(spec)
    metricas_precisao.append(prec)
    metricas_f1_score.append(f1)
    
    if (r + 1) % 50 == 0:
        print(f"Rodada {r + 1}/{R} concluída.")

plt.show()
plt.show(block=True)        
print("\nSimulação concluída.")

Avaliador.print_stat("Acurácia", metricas_acuracia)
Avaliador.print_stat("Sensibilidade", metricas_sensibilidade)
Avaliador.print_stat("Especificidade", metricas_especificidade)
Avaliador.print_stat("Precisão", metricas_precisao)
Avaliador.print_stat("F1-Score", metricas_f1_score)



plt.ioff()
plt.show()