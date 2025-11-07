import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import sys
import os
import cv2  # Import do OpenCV

# --- Caminhos ---
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Avaliador import Avaliador
from Matriz_Confusao_MLP import Matriz_Confusao

from neural_network import MultilayerPerceptron


# ----------------------------------------------------------
# Função auxiliar para carregar dados (OpenCV)
# (Sem alterações)
# ----------------------------------------------------------
def carregar_imagens_recfac(root_folder, size=(15, 15)):
    """
    Carrega imagens da base RecFac usando OpenCV.
    Converte para escala de cinza, redimensiona e achata.
    """
    X_list, labels = [], []
    valid_ext = {'.png'} # Ajuste se tiver .jpg, .jpeg, etc.

    # Verifica se o diretório raiz existe
    if not os.path.isdir(root_folder):
        raise RuntimeError(f"Diretório não encontrado: {root_folder}")

    for root, _, files in os.walk(root_folder):
        for fname in files:
            _, ext = os.path.splitext(fname)
            if ext.lower() not in valid_ext:
                continue
            
            path = os.path.join(root, fname)
            # Pega o nome da pasta pai como label (ex: 'pessoa1')
            person = os.path.basename(os.path.dirname(path))

            # Carrega em escala de cinza
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"[AVISO] Não foi possível ler: {path}")
                continue
            
            # Redimensiona para o tamanho padrão
            img = cv2.resize(img, size)
            
            # vetoor
            X_list.append(img.flatten().astype(np.float32))
            labels.append(person)

    if not X_list:
        raise RuntimeError(f"Nenhuma imagem {valid_ext} foi encontrada em {root_folder}/.")
    
    X_all = np.vstack(X_list)
    y_labels = np.array(labels)
    return X_all, y_labels

# ----------------------------------------------------------
# CARREGAMENTO DOS DADOS (OpenCV - RecFac)
# ----------------------------------------------------------
print("Carregando imagens da base RecFac...")
# O caminho 'RecFac/' funciona pois está no mesmo nível do main.py
ROOT_FOLDER = os.path.join(parent_dir, "RecFac") 

try:
    X_all, y_labels_all = carregar_imagens_recfac(ROOT_FOLDER, size=(15, 15))
except RuntimeError as e:
    print(e)
    print(f"ERRO: Verifique se a pasta '{ROOT_FOLDER}' existe e contém subpastas com imagens .png.")
    sys.exit(1)

# ----------------------------------------------------------
# ADAPTAÇÃO PARA CLASSIFICAÇÃO MULTICLASSE
# ----------------------------------------------------------
unique_labels = np.unique(y_labels_all)
n_classes = len(unique_labels)

print(f"Detectadas {n_classes} classes: {unique_labels}")

# Criação de mapeamento label → índice
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
y_idx = np.array([label_to_idx[l] for l in y_labels_all])

# Codificação one-hot (para MLP)
Y_one_hot = np.zeros((y_idx.size, n_classes))
Y_one_hot[np.arange(y_idx.size), y_idx] = 1

X = X_all
y = Y_one_hot 

if X.shape[0] == 0:
     print(f"ERRO: Nenhum dado encontrado para as classes {label_1} e {label_minus_1}.")
     sys.exit(1)

print(f"Total de {X.shape[0]} amostras (vetores de {X.shape[1]} features) carregadas.")

N, p = X.shape
# O plot inicial (scatter 2D) não se aplica a dados de imagem (alta dimensão)

# ----------------------------------------------------------
# SIMULAÇÃO DE MONTE CARLO
# (Sem alterações)
# ----------------------------------------------------------

metricas_acuracia = []
metricas_sensibilidade = []
metricas_especificidade = []
metricas_precisao = []
metricas_f1_score = []
resultados = []
R = 1 #TODO ajustar

print(f"\nIniciando simulação de Monte Carlo com {R} rodadas...")
for r in range(R):
    
    idx = np.random.permutation(N)
    Xr = X[idx,:]
    yr = y[idx, :]
    
    # Particionamento (80% treino, 20% teste)
    split_idx = int(N * 0.8)
    X_treino = Xr[:split_idx, :]
    y_treino = yr[:split_idx, :]
    
    X_teste = Xr[split_idx:, :] 
    y_teste = yr[split_idx:, :]
    
    # NORMALIZAÇÃO DOS DADOS [-1, 1] 
    min_treino = X_treino.min(axis=0)
    max_treino = X_treino.max(axis=0)
    denominador = (max_treino - min_treino) + 1e-8
    X_treino_norm = 2 * (X_treino - min_treino) / denominador - 1
    X_teste_norm = 2 * (X_teste - min_treino) / denominador - 1

    # FIM DA NORMALIZAÇÃO
    
    layer_dims = [X_treino.shape[1], 30]
    
    ps = MultilayerPerceptron(topology=layer_dims, X_train=X_treino_norm.T, Y_train=y_treino.T, max_epoch=100, learning_rate=0.01)
    ps.fit()
      
    y_pred = ps.predict(X_teste_norm.T)
    y_pred_bin = np.where(y_pred >= 0, 1, -1)
    
    acc, sens, spec, prec, f1 = Avaliador.calcular_metricas(y_teste, y_pred_bin)

    metricas_acuracia.append(acc)
    metricas_sensibilidade.append(sens)
    metricas_especificidade.append(spec)
    metricas_precisao.append(prec)
    metricas_f1_score.append(f1)
    
    resultados.append({
        "acc": acc, "sens": sens, "spec": spec, "prec": prec, "f1": f1,
        "y_true": y_teste.flatten(),
        "y_pred": y_pred.flatten(), # Salva a saída real (antes de binarizar)
        "errors": ps.errors_per_epoch
    })
    
    if (r + 1) % 50 == 0 or (r + 1) == R:
        print(f"Rodada {r + 1}/{R} concluída.")

# ----------------------------------------------------------
# ----- PLOTAGEM (Apenas Acurácia) -----
# (Sem alterações na lógica de plotagem)
# ----------------------------------------------------------

metricas = ["acc"] # <-- Focado apenas na Acurácia, conforme pedido
labels_plot = [f'{i} ({label})' for i, label in enumerate(unique_labels)]


# Definição das cores
GREENS = ['#006045', '#009966', '#00d492', '#a4f4cf']
REDS = ['#a50036', '#ec003f', '#ff637e', '#ffccd3']
cmap_greens = ListedColormap(GREENS)
cmap_reds   = ListedColormap(REDS)
mc_indices = np.array([[0, 1], [2, 3]])

for metrica in metricas:
    
    # 1. Encontrar melhor e pior rodada
    melhor = Avaliador.get_melhor(metrica, resultados)
    pior = Avaliador.get_pior(metrica, resultados)

    # 2. Preparar dados para Matriz de Confusão
    y_true_melhor = melhor["y_true"].ravel()
    y_pred_melhor = np.where(melhor["y_pred"].ravel() >= 0, 1, -1)
    
    y_true_pior = pior["y_true"].ravel()
    y_pred_pior = np.where(pior["y_pred"].ravel() >= 0, 1, -1)

    mc_melhor = Matriz_Confusao.conf_matriz(y_true_melhor, y_pred_melhor)
    mc_pior = Matriz_Confusao.conf_matriz(y_true_pior, y_pred_pior)
    
    # 3. Plotar Matrizes de Confusão (Melhor x Pior)
    fig_cm, (ax_cm_melhor, ax_cm_pior) = plt.subplots(1, 2, figsize=(16, 7))
    
    labels_plot = np.unique(np.concatenate((y_true_melhor, y_pred_melhor)))
    sns.heatmap(mc_indices, annot=mc_melhor, fmt='d',
            cmap=cmap_greens, cbar=False,
            xticklabels=labels_plot, yticklabels=labels_plot)
    ax_cm_melhor.set_title(f'Melhor {metrica.upper()} - Matriz de Confusão')
    ax_cm_melhor.set_xlabel('Predito (Previsto)')
    ax_cm_melhor.set_ylabel('Verdadeiro (Real)')
    ax_cm_melhor.set_yticklabels(ax_cm_melhor.get_yticklabels(), rotation=0)

    # sns.heatmap(mc_indices, annot=mc_pior, fmt='d',
    #             cmap=cmap_reds, cbar=False, ax=ax_cm_pior,
    #             xticklabels=labels_plot, yticklabels=labels_plot)
    # ax_cm_pior.set_title(f'Pior {metrica.upper()} - Matriz de Confusão')
    # ax_cm_pior.set_xlabel('Predito (Previsto)')
    # ax_cm_pior.set_ylabel('Verdadeiro (Real)')
    # ax_cm_pior.set_yticklabels(ax_cm_pior.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.show()
    
    # 4. Preparar Curvas de Aprendizado
    errors_melhor = melhor["errors"] if melhor["errors"] is not None else [0]
    errors_pior = pior["errors"] if pior["errors"] is not None else [0]
    
    # 5. Plotar Curvas de Aprendizado (Melhor x Pior)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(errors_melhor, color=GREENS[1], lw=2)
    axes[0].set_title(f"Melhor {metrica.upper()}")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Erros por época")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].set_yscale('log')

    axes[1].plot(errors_pior, color=REDS[1], lw=2)
    axes[1].set_title(f"Pior {metrica.upper()}")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Erros por época")
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].set_yscale('log')
    
    plt.suptitle(f"Curvas de Aprendizado - {metrica.upper()}", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# ----------------------------------------------------------
# --- Finalização (Imprime estatísticas de todas as métricas) ---
# ----------------------------------------------------------
print("\nSimulação concluída.")
print("Estatísticas gerais (todas as rodadas):")
Avaliador.print_stat("Acurácia", metricas_acuracia)
Avaliador.print_stat("Sensibilidade", metricas_sensibilidade)
Avaliador.print_stat("Especificidade", metricas_especificidade)
Avaliador.print_stat("Precisão", metricas_precisao)
Avaliador.print_stat("F1-Score", metricas_f1_score)

# plt.show() # Garante que todos os plots abertos sejam exibidos