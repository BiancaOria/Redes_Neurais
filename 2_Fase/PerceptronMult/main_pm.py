import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import sys
import os
import cv2  

# --- Caminhos ---
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Avaliador_MLP import Avaliador
from Matriz_Confusao_MLP import Matriz_Confusao

from neural_network import MultilayerPerceptron


# ----------------------------------------------------------
# Função auxiliar para carregar dados (OpenCV)
# (Sem alterações)
# ----------------------------------------------------------
def carregar_imagens_recfac(root_folder, size=(30, 30)):
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
    X_all, y_labels_all = carregar_imagens_recfac(ROOT_FOLDER, size=(30, 30))
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
    
    layer_dims = [X_treino.shape[1], 500, 300, 100, 50]
    
    ps = MultilayerPerceptron(topology=layer_dims, X_train=X_treino_norm.T, Y_train=y_treino.T, max_epoch=100, learning_rate=0.01)
    ps.fit()
    
    y_pred_scores = ps.predict(X_teste_norm.T) # Saída é (n_classes, n_samples)

    y_pred_scores = np.squeeze(y_pred_scores)  # remove dimensões extras
    if y_pred_scores.ndim == 1:
        y_pred_scores = y_pred_scores[np.newaxis, :]  # garante 2D
    elif y_pred_scores.shape[0] != y_teste.shape[0]:
        y_pred_scores = y_pred_scores.T  # ajusta orientação se necessário

    print("y_pred_scores:", y_pred_scores.shape)
    print("y_teste:", y_teste.shape)


    y_pred_indices = np.argmax(y_pred_scores, axis=1)
    y_teste_indices = np.argmax(y_teste, axis=1)

    # ATENÇÃO: Seu 'Avaliador' provavelmente também está binário.
    # As métricas (acc, sens, etc.) aqui podem estar erradas para multiclasse.
    # O código abaixo é mantido para não quebrar, mas foca na matriz de confusão.
    y_pred_bin = np.where(y_pred_scores >= 0, 1, -1)
    acc = Avaliador.calcular_metricas(y_teste_indices, y_pred_indices) # Isso ainda compara (one-hot) com (binário)

    metricas_acuracia.append(acc) # TODO: Idealmente, 'acc' deveria ser recalculada
   
    
    resultados.append({
        "acc": acc,
        # --- !! MUDANÇA CRÍTICA !! ---
        # Salvar os ÍNDICES, não os vetores/scores achatados
        "y_true": y_teste_indices, # AGORA é (n_samples,)
        "y_pred": y_pred_indices, # AGORA é (n_samples,)
        # --- !! FIM DA MUDANÇA !! ---
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
    # Os dados já estão salvos como índices (0, 1, 2...)
    y_true_melhor = melhor["y_true"]
    y_pred_melhor = melhor["y_pred"]
    
    y_true_pior = pior["y_true"]
    y_pred_pior = pior["y_pred"]

    # Passamos os labels de índice (0, 1, ..., n_classes-1) para garantir a ordem
    labels_indices = list(range(n_classes))

    # A função agora retorna a matriz E os labels que ela usou
    # Gera matriz N×N e também reduzida 2×2
    mc_melhor_full, _ = Matriz_Confusao.conf_matriz(y_true_melhor, y_pred_melhor, labels=labels_indices)
    mc_pior_full, _   = Matriz_Confusao.conf_matriz(y_true_pior, y_pred_pior, labels=labels_indices)

    # Matriz 2×2 agregada (VP, VN, FP, FN)
    mc_melhor, _ = Matriz_Confusao.conf_matriz(y_true_melhor, y_pred_melhor, labels=labels_indices, binarizar=True)
    mc_pior, _   = Matriz_Confusao.conf_matriz(y_true_pior, y_pred_pior, labels=labels_indices, binarizar=True)
    
    # 3. Plotar Matrizes 2x2 (Melhor x Pior)
    fig_cm, (ax_cm_melhor, ax_cm_pior) = plt.subplots(1, 2, figsize=(10, 5))

    # Labels da 2x2
    labels_bin = ['Positivo', 'Negativo']

    mc_indices = np.array([[0, 1],[2, 3]])

    sns.heatmap(mc_indices, annot=mc_melhor, fmt='d', cmap=cmap_greens, cbar=False,
                xticklabels=labels_bin, yticklabels=labels_bin, ax=ax_cm_melhor)
    ax_cm_melhor.set_title(f'Melhor {metrica.upper()}')
    ax_cm_melhor.set_xlabel('Predito')
    ax_cm_melhor.set_ylabel('Verdadeiro')

    sns.heatmap(mc_indices, annot=mc_pior, fmt='d', cmap=cmap_reds, cbar=False,
                xticklabels=labels_bin, yticklabels=labels_bin, ax=ax_cm_pior)
    ax_cm_pior.set_title(f'Pior {metrica.upper()}')
    ax_cm_pior.set_xlabel('Predito')
    ax_cm_pior.set_ylabel('Verdadeiro')

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


