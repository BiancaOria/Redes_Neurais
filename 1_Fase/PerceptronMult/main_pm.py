import numpy as np
import matplotlib.pyplot as plt
from Perceptron_Mult import Perceptron_Mult
import seaborn as sns
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Avaliador import Avaliador
from Matriz_Confusao import Matriz_Confusao


data = np.loadtxt("../spiral_d.csv", delimiter=',')


X = data[:, :-1]
y = data[:, -1:]
N, p = X.shape
d = np.array(y).flatten()
XT=X.T

#plot incial
fig = plt.figure(1)
ax = fig.add_subplot()
ax.scatter(XT[0, (d==1)],
                XT[1, (d==1)],
                c='r', marker='s', s=120, edgecolor='k')

ax.scatter(XT[0, (d==-1)],
                XT[1, (d==-1)],
                c='b', marker='o', s=120, edgecolor='k')
margin = 1  # margem extra
x_min, x_max = XT[0].min() - margin, XT[0].max() + margin
y_min, y_max = XT[1].min() - margin, XT[1].max() + margin


ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)


ax.set_xlabel("Variável 1")
ax.set_ylabel("Variável 2")

ax.set_title("Visualização Inicial dos Dados")
# ax.legend()
ax.grid(True)

#monte carlo

metricas_acuracia = []
metricas_sensibilidade = []
metricas_especificidade = []
metricas_precisao = []
metricas_f1_score = []
resultados = []
R = 1 #ajustar

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
    
    # NORMALIZAÇÃO DOS DADOS [-1, 1] 
    min_treino = X_treino.min(axis=0)
    max_treino = X_treino.max(axis=0)
    
    denominador = (max_treino - min_treino) + 1e-8
    
    # x_norm = 2 * (x - min) / (max - min) - 1 ) 
    X_treino_norm = 2 * (X_treino - min_treino) / denominador - 1
    
    # Normalizar o X_teste com base no que foi obtido no treino
    X_teste_norm = 2 * (X_teste - min_treino) / denominador - 1
 
    # FIM DA NORMALIZAÇÃO
    
    layer_dims = [X_treino.shape[1], 2, 5, 1]
    ps = Perceptron_Mult(layer_dims, X_treino_norm.T, y_treino, plot=True, max_epoch=100,learning_rate=0.01)
    ps.fit()
    
    
    
    y_pred = ps.predict(X_teste_norm.T)
    
    acc, sens, spec, prec, f1 = Avaliador.calcular_metricas(y_teste, y_pred)
    
    metricas_acuracia.append(acc)
    metricas_sensibilidade.append(sens)
    metricas_especificidade.append(spec)
    metricas_precisao.append(prec)
    metricas_f1_score.append(f1)
    resultados.append({
        "acc": acc, "sens": sens, "spec": spec, "prec": prec, "f1": f1,
        "y_true": y_teste.flatten(), "y_pred": y_pred.flatten(),
        "errors": ps.errors_per_epoch
    })
    
    if (r + 1) % 50 == 0:
        print(f"Rodada {r + 1}/{R} concluída.")
        
# ----- MATRIZ DE CONFUSÃO -----
metricas = ["acc", "sens", "spec", "prec", "f1"]
labels_plot = ['Classe 1', 'Classe -1']

for metrica in metricas:
    
    melhor= Avaliador.get_melhor(metrica,resultados)
    pior = Avaliador.get_pior(metrica,resultados)
    
    mc_melhor = Matriz_Confusao.conf_matriz(melhor["y_true"], melhor["y_pred"])
    mc_pior = Matriz_Confusao.conf_matriz(pior["y_true"], pior["y_pred"])
    
    # 3. Criar a figura com 2 subplots (1 linha, 2 colunas)
    fig_cm, (ax_cm_melhor, ax_cm_pior) = plt.subplots(1, 2, figsize=(14, 6))
    
    GREENS = ['#006045', '#009966', '#00d492', '#a4f4cf']

    REDS = ['#a50036', '#ec003f', '#ff637e', '#ffccd3']
    
    # Cria colormaps fixos
    cmap_greens = ListedColormap(GREENS)
    cmap_reds   = ListedColormap(REDS)

    # Normaliza os valores da matriz pra índices 0–3
    # Assim cada quadrado pega uma cor específica
    mc_indices = np.array([[0, 1],
                        [2, 3]])

    # Plotar "Melhor"
    sns.heatmap(mc_indices, annot=mc_melhor, fmt='d',
                cmap=cmap_greens, cbar=False, ax=ax_cm_melhor,
                xticklabels=labels_plot, yticklabels=labels_plot)
    ax_cm_melhor.set_title(f'Melhor {metrica.upper()} - Matriz de Confusão')
    ax_cm_melhor.set_xlabel('Predito (Previsto)')
    ax_cm_melhor.set_ylabel('Verdadeiro (Real)')
    ax_cm_melhor.set_yticklabels(ax_cm_melhor.get_yticklabels(), rotation=0)

    # Plotar "Pior"
    sns.heatmap(mc_indices, annot=mc_pior, fmt='d',
                cmap=cmap_reds, cbar=False, ax=ax_cm_pior,
                xticklabels=labels_plot, yticklabels=labels_plot)
    ax_cm_pior.set_title(f'Pior {metrica.upper()} - Matriz de Confusão')
    ax_cm_pior.set_xlabel('Predito (Previsto)')
    ax_cm_pior.set_ylabel('Verdadeiro (Real)')
    ax_cm_pior.set_yticklabels(ax_cm_pior.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.show() # Mostra a figura das matrizes
    
    # ----- CURVA DE APRENDIZADO -----
    
    # Garante que 'errors' não é None
    errors_melhor = melhor["errors"] if melhor["errors"] is not None else [0]
    errors_pior = pior["errors"] if pior["errors"] is not None else [0]
    # --- CURVAS DE APRENDIZADO (Melhor x Pior) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # 1 linha, 2 colunas

    # --- Subgráfico 1: Melhor ---
    axes[0].plot(errors_melhor, color='green')
    axes[0].set_title(f"Melhor {metrica.upper()}")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Erros por época")
    axes[0].grid(True)

    # --- Subgráfico 2: Pior ---
    axes[1].plot(errors_pior, color='red')
    axes[1].set_title(f"Pior {metrica.upper()}")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Erros por época")
    axes[1].grid(True)

    # Ajustes finais
    plt.suptitle(f"Curvas de Aprendizado - {metrica.upper()}", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Deixa espaço para o título
    plt.show()

# --- Finalização (sem alterações) ---
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