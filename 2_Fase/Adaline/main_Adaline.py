import cv2
from Adaline import ADALINE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Avaliador import Avaliador
from Matriz_Confusao import Matriz_Confusao


# ----------------------------------------------------------
# Funções auxiliares
# ----------------------------------------------------------
def carregar_imagens_recfac(root_folder, size=(40, 40)):
    """
    Lê todas as imagens da pasta RecFac (e subpastas com nomes de pessoas).
    Retorna:
        X -> array (N, p)
        labels -> lista com o nome de cada pessoa
    """
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
    """
    Retorna Y (N, C) em codificação bipolar (+1/-1)
    """
    classes = sorted(list(set(labels)))
    c_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    Y = -1 * np.ones((len(labels), n_classes), dtype=int)
    for i, label in enumerate(labels):
        Y[i, c_to_idx[label]] = 1
    return Y, c_to_idx, classes


def decodificar_one_hot_bipolar(Y_one_hot):
    """
    Converte uma matriz one-hot bipolar (+1 / -1)
    em um vetor de índices de classes.

    Exemplo:
        Entrada:
            [[-1, +1, -1],
             [+1, -1, -1],
             [-1, -1, +1]]
        Saída:
            [1, 0, 2]
    """
    return np.argmax(Y_one_hot, axis=1)




# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
if __name__ == "__main__":
    ROOT = os.path.join(parent_dir, "RecFac")
    IMG_SIZE = (40, 40)
    R = 1  # TODO ALTERAR
    LR = 0.001
    MAX_EPOCH = 300
    TOL = 1e-6

    print("Carregando imagens da pasta RecFac...")
    X_raw, labels = carregar_imagens_recfac(ROOT, size=IMG_SIZE)
    N, p = X_raw.shape
    print(f"Total de imagens: {N}")
    print(f"Tamanho das imagens: {IMG_SIZE[0]}x{IMG_SIZE[1]} = {p} neurônios de entrada")

    # One-hot bipolar
    Y, label_to_idx, classes = codificar_one_hot_bipolar(labels)
    n_classes = len(classes)
    y_idx = np.array([label_to_idx[l] for l in labels])
    print(f"Classes detectadas ({n_classes}): {list(range(n_classes))}")

    # Normalização [-1, 1]
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

        valores_idx = []

        # Treina 1 ADALINE por classe (One-vs-Rest)
        models = []
        for c in range(n_classes):
            y_c = Y_train[:, c].reshape(-1, 1)
            model = ADALINE(X_train_T, y_c, learning_rate=LR, max_epoch=MAX_EPOCH, tol=TOL)
            model.fit()
            models.append(model)

        # Predição
        ativacoes = np.zeros((n_classes, X_test.shape[0]))
        for c, model in enumerate(models):
            ativacoes[c, :] = model.predict_raw(X_test_T)

        # Índice da maior ativação
        idx_pred = np.argmax(ativacoes, axis=0)

        valores_idx.push(idx_pred)

        # Converte para vetor one-hot bipolar (+1 / -1)
        Y_pred = -1 * np.ones((X_test.shape[0], n_classes), dtype=int)
        for i, c_idx in enumerate(idx_pred):
            Y_pred[i, c_idx] = 1

        y_pred = idx_pred

        acuracia = np.mean(y_pred == y_true_test)
        accs.append(acuracia)

        # Curva de aprendizado média
        max_len = max(len(m.errors_per_epoch) for m in models)
        eqms = np.zeros((n_classes, max_len))
        for c, m in enumerate(models):
            e = np.array(m.errors_per_epoch)
            if len(e) < max_len:
                e = np.concatenate([e, [e[-1]] * (max_len - len(e))])
            eqms[c, :] = e
        mean_eqm = eqms.mean(axis=0)

        resultados.append({
            "acc": acuracia,
            "y_true": y_true_test,
            "y_pred": y_pred,
            "Y_pred": Y_pred, 
            "mean_eqm": mean_eqm
        })

        print(f"Rodada {r+1}/{R} concluída. Acurácia = {acuracia*100:.2f}%")

    # Estatísticas
    accs = np.array(accs)
    best_idx = np.argmax(accs)
    worst_idx = np.argmin(accs)
    best = resultados[best_idx]
    worst = resultados[worst_idx]

    print("\nResumo das 10 rodadas:")
    print(f"Acurácia média: {accs.mean()*100:.2f}%")
    print(f"Desvio-padrão: {accs.std()*100:.2f}%")
    print(f"Melhor rodada ({best_idx+1}): {accs[best_idx]*100:.2f}%")
    print(f"Pior rodada ({worst_idx+1}): {accs[worst_idx]*100:.2f}%")

    # Matrizes de confusão
    cm_best = Matriz_Confusao(best["y_true"], best["y_pred"], n_classes)
    cm_worst = Matriz_Confusao(worst["y_true"], worst["y_pred"], n_classes)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(cm_best, annot=True, fmt='d', ax=ax1, cbar=False)
    ax1.set_title(f"Melhor rodada ({best_idx+1}) - Acurácia {accs[best_idx]*100:.2f}%")
    sns.heatmap(cm_worst, annot=True, fmt='d', ax=ax2, cbar=False)
    ax2.set_title(f"Pior rodada ({worst_idx+1}) - Acurácia {accs[worst_idx]*100:.2f}%")
    plt.show()

    # Curvas de aprendizado
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(best["mean_eqm"], color='green')
    plt.title("Curva de Aprendizado - Melhor Rodada")
    plt.xlabel("Época")
    plt.ylabel("EQM médio")

    plt.subplot(1, 2, 2)
    plt.plot(worst["mean_eqm"], color='red')
    plt.title("Curva de Aprendizado - Pior Rodada")
    plt.xlabel("Época")
    plt.ylabel("EQM médio")

    plt.tight_layout()
    plt.show()

    print("\nSimulação concluída.")
