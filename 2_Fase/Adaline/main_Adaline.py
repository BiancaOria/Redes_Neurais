import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import sys
import os

# --- Caminhos ---
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Adaline import ADALINE
from Avaliador import Avaliador
from Matriz_Confusao import Matriz_Confusao


# ----------------------------------------------------------
# Funções auxiliares
# ----------------------------------------------------------
def carregar_imagens_recfac(root_folder, size=(40, 40)):
    X_list, labels = [], []
    valid_ext = {'.png'}

    for root, _, files in os.walk(root_folder):
        for fname in files:
            _, ext = os.path.splitext(fname)
            if ext.lower() not in valid_ext:
                continue
            path = os.path.join(root, fname)
            person = os.path.basename(os.path.dirname(path))

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[AVISO] Não foi possível ler: {path}")
                continue
            img = cv2.resize(img, size)
            X_list.append(img.flatten().astype(np.float32))
            labels.append(person)

    if not X_list:
        raise RuntimeError("Nenhuma imagem foi encontrada em RecFac/.")
    X = np.vstack(X_list)
    return X, labels


def codificar_one_hot_bipolar(labels):
    classes = sorted(list(set(labels)))
    c_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    Y = -1 * np.ones((len(labels), n_classes), dtype=int)
    for i, label in enumerate(labels):
        Y[i, c_to_idx[label]] = 1
    return Y, c_to_idx, classes


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
if __name__ == "__main__":
    ROOT = os.path.join(parent_dir, "RecFac")
    IMG_SIZE = (40, 40)
    R = 1 #TODO ajustar p 10 ou 100
    LR = 0.001
    MAX_EPOCH = 300
    TOL = 1e-6

    print("Carregando imagens da pasta RecFac...")
    X_raw, labels = carregar_imagens_recfac(ROOT, size=IMG_SIZE)
    N, p = X_raw.shape
    print(f"Total de imagens: {N}")
    print(f"Tamanho das imagens: {IMG_SIZE[0]}x{IMG_SIZE[1]} = {p} neurônios de entrada")

    Y, label_to_idx, classes = codificar_one_hot_bipolar(labels)
    n_classes = len(classes)
    y_idx = np.array([label_to_idx[l] for l in labels])
    print(f"Classes detectadas ({n_classes}): {list(range(n_classes))}")

    X_norm = (X_raw / 127.5) - 1.0

    resultados = []
    accs = []

    print(f"\nIniciando Monte Carlo com R = {R} rodadas...\n")
    for r in range(R):
        idx = np.random.permutation(N)
        Xr = X_norm[idx, :]
        Yr = Y[idx, :]
        y_idx_r = y_idx[idx]

        split = int(0.8 * N)
        X_train, X_test = Xr[:split], Xr[split:]
        Y_train, Y_test = Yr[:split], Yr[split:]
        y_true_test = y_idx_r[split:]

        X_train_T = X_train.T
        X_test_T = X_test.T

        models = []
        for c in range(n_classes):
            y_c = Y_train[:, c].reshape(-1, 1)
            model = ADALINE(X_train_T, y_c, learning_rate=LR, max_epoch=MAX_EPOCH, tol=TOL, plot=False)
            model.fit()
            models.append(model)

        ativacoes = np.zeros((n_classes, X_test.shape[0]))
        for c, model in enumerate(models):
            ativacoes[c, :] = np.dot(model.w.T, np.vstack((-np.ones((1, X_test_T.shape[1])), X_test_T)))

        idx_pred = np.argmax(ativacoes, axis=0)
        Y_pred = -1 * np.ones((X_test.shape[0], n_classes), dtype=int)
        for i, c_idx in enumerate(idx_pred):
            Y_pred[i, c_idx] = 1

        y_pred = idx_pred
        acuracia = np.mean(y_pred == y_true_test)
        accs.append(acuracia)

        resultados.append({
            "acc": acuracia,
            "models": models,
            "Y_test": Y_test,
            "Y_pred": Y_pred
        })
        print(f"Rodada {r+1}/{R} concluída. Acurácia = {acuracia*100:.2f}%")

    # Selecionar melhor e pior rodada
    accs = np.array(accs)
    best_idx = np.argmax(accs)
    worst_idx = np.argmin(accs)
    best = resultados[best_idx]
    worst = resultados[worst_idx]

    print(f"\nMelhor rodada ({best_idx+1}) -> Acurácia: {accs[best_idx]*100:.2f}%")
    print(f"Pior rodada ({worst_idx+1}) -> Acurácia: {accs[worst_idx]*100:.2f}%")

    # ----------------------------------------------------------
    # MATRIZES 2×2 INDIVIDUAIS - MELHOR E PIOR
    # ----------------------------------------------------------

    GREENS = ['#006045', '#009966', '#00d492', '#a4f4cf']
    REDS = ['#a50036', '#ec003f', '#ff637e', '#ffccd3']
    cmap_greens = ListedColormap(GREENS)
    cmap_reds = ListedColormap(REDS)

    def gerar_matrizes(Y_true, Y_pred, cmap, titulo):
        fig, axes = plt.subplots(4, 5, figsize=(18, 12))
        axes = axes.flatten()
        soma_total = 0

        for c in range(n_classes):
            y_true_c = Y_true[:, c]
            y_pred_c = Y_pred[:, c]

            # Normaliza os valores da matriz pra índices 0–3
            # Assim cada quadrado pega uma cor específica
            mc_indices = np.array([[0, 1],
                                [2, 3]])

            mc = Matriz_Confusao.conf_matriz(y_true_c, y_pred_c, labels=[1, -1])
            soma_total += mc.sum()

            sns.heatmap(mc_indices, annot=mc, fmt='d', cmap=cmap, cbar=False,
                        xticklabels=[1, -1], yticklabels=[1, -1], ax=axes[c])
            axes[c].set_title(f"Classe {c+1}", fontsize=10)
            axes[c].set_xlabel("Predito")
            axes[c].set_ylabel("Verdadeiro")

        plt.suptitle(f"Matrizes de Confusão - {titulo}", fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        print(f"Soma total ({titulo}): {soma_total} (deve ser 640)")

    # --- Plotar as 20 melhores (verde) e 20 piores (vermelho)
    gerar_matrizes(best["Y_test"], best["Y_pred"], cmap_greens, "Melhor Rodada")
    gerar_matrizes(worst["Y_test"], worst["Y_pred"], cmap_reds, "Pior Rodada")

    print("\nSimulação concluída com sucesso.")
