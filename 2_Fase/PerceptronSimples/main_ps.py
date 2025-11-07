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

from Perceptron import Perceptron
from Avaliador import Avaliador
from Matriz_Confusao import Matriz_Confusao


# ----------------------------------------------------------
# Funções auxiliares
# ----------------------------------------------------------
def carregar_imagens_recfac(root_folder, size=(30, 30)):
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
    IMG_SIZE = (30, 30)
    R = 10 #TODO ajustar p 10 
    LR = 0.001
    MAX_EPOCH = 300

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
        X_treino, X_teste = Xr[:split], Xr[split:]
        Y_train, Y_test = Yr[:split], Yr[split:]
        y_true_test = y_idx_r[split:]

        
        # NORMALIZAÇÃO DOS DADOS [-1, 1] 
        min_treino = X_treino.min(axis=0)
        max_treino = X_treino.max(axis=0)
        
        denominador = (max_treino - min_treino) + 1e-8
        
        # x_norm = 2 * (x - min) / (max - min) - 1 ) 
        X_treino_norm = 2 * (X_treino - min_treino) / denominador - 1
        
        # Normalizar o X_teste com base no que foi obtido no treino
        X_teste_norm = 2 * (X_teste - min_treino) / denominador - 1

        # FIM DA NORMALIZAÇÃO

        models = []
        for c in range(n_classes):
            y_c = Y_train[:, c].reshape(-1, 1)
            model = Perceptron(X_treino_norm.T, y_c, learning_rate=LR, max_epoch=MAX_EPOCH, plot=False)
            model.fit()
            models.append(model)

        ativacoes = np.zeros((n_classes, X_teste_norm.shape[0]))
        for c, model in enumerate(models):
            ativacoes[c, :] = np.dot(model.w.T, np.vstack((-np.ones((1, X_teste_norm.T.shape[1])), X_teste_norm.T)))

        idx_pred = np.argmax(ativacoes, axis=0)
        Y_pred = -1 * np.ones((X_teste.shape[0], n_classes), dtype=int)
        for i, c_idx in enumerate(idx_pred):
            Y_pred[i, c_idx] = 1

        y_pred = idx_pred
        acuracia = np.mean(y_pred == y_true_test)
        accs.append(acuracia)
        # print("--"*20)
        # print(y_pred)
        # print("--"*20)
        # print(y_true_test)
        # print("--"*20)
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

        # print(f"Soma total ({titulo}): {soma_total} (deve ser 640)") 

    # --- Plotar as 20 melhores (verde) e 20 piores (vermelho)
    gerar_matrizes(best["Y_test"], best["Y_pred"], cmap_greens, "Melhor Rodada")
    gerar_matrizes(worst["Y_test"], worst["Y_pred"], cmap_reds, "Pior Rodada")

    # ----------------------------------------------------------
    # FUNÇÃO AUXILIAR PARA CURVAS DE APRENDIZADO
    # ----------------------------------------------------------
    import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import sys
import os
import cv2  # Import do OpenCV
# Import necessário para One-Hot Encoding
from sklearn.preprocessing import OneHotEncoder

# --- Caminhos (Preservados) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Avaliador import Avaliador
from Matriz_Confusao import Matriz_Confusao
# Import da MLP (Preservado)
from neural_network import MultilayerPerceptron


# ----------------------------------------------------------
# Função auxiliar para carregar dados (OpenCV)
# (Alteração 'size=(10, 10)' preservada)
# ----------------------------------------------------------
def carregar_imagens_recfac(root_folder, size=(10, 10)):
    """
    Carrega imagens da base RecFac usando OpenCV.
    Converte para escala de cinza, redimensiona e achata.
    """
    X_list, labels = [], []
    valid_ext = {'.png'} 

    if not os.path.isdir(root_folder):
        raise RuntimeError(f"Diretório não encontrado: {root_folder}")

    for root, _, files in os.walk(root_folder):
        for fname in files:
            _, ext = os.path.splitext(fname)
            if ext.lower() not in valid_ext:
                continue
            
            path = os.path.join(root, fname)
            person = os.path.basename(os.path.dirname(path)) # Label (ex: 'pessoa1')

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"[AVISO] Não foi possível ler: {path}")
                continue
            
            img = cv.resize(img, size) # Redimensiona (10x10)
            
            X_list.append(img.flatten().astype(np.float32)) # Achata (vetor de 100)
            labels.append(person)

    if not X_list:
        raise RuntimeError(f"Nenhuma imagem {valid_ext} foi encontrada em {root_folder}/.")
    
    X_all = np.vstack(X_list)
    y_labels = np.array(labels)
    return X_all, y_labels

# ----------------------------------------------------------
# CARREGAMENTO DOS DADOS (OpenCV - RecFac)
# (Caminho 'ROOT_FOLDER' preservado)
# ----------------------------------------------------------
print("Carregando imagens da base RecFac...")
ROOT_FOLDER = os.path.join(parent_dir, "RecFac") 

try:
    X_all, y_labels_all = carregar_imagens_recfac(ROOT_FOLDER, size=(10, 10))
except RuntimeError as e:
    print(e)
    print(f"ERRO: Verifique se a pasta '{ROOT_FOLDER}' existe e contém subpastas com imagens .png.")
    sys.exit(1)

# ----------------------------------------------------------
# PREPARAÇÃO MULTI-CLASSE (20 Classes)
# (Substitui o bloco de adaptação binária)
# ----------------------------------------------------------

unique_labels = np.unique(y_labels_all)
n_classes = len(unique_labels)

print(f"Encontradas {n_classes} classes únicas.")

# Validação (baseado na 'layer_dims' que você forneceu)
if n_classes != 20:
    print(f"[AVISO] O script encontrou {n_classes} classes, mas a topologia da rede")
    print(f"         definida em 'layer_dims' espera 20 classes (neurônios) na saída.")
    print(f"         Continuando, mas isso pode causar um erro se 'n_classes' for > 20.")

# 1. Mapear labels (string) para inteiros (0 a 19)
label_to_int = {label: i for i, label in enumerate(unique_labels)}
y_int = np.array([label_to_int[label] for label in y_labels_all])

# 2. Converter inteiros para One-Hot Encoding
# Ex: Classe 3 (de 20) -> [0, 0, 0, 1, 0, ..., 0]
encoder = OneHotEncoder(sparse_output=False, categories='auto')
# 'y' agora tem shape (N_amostras, 20)
y = encoder.fit_transform(y_int.reshape(-1, 1))

# 'X' são todas as amostras
X = X_all

print(f"Total de {X.shape[0]} amostras (vetores de {X.shape[1]} features) carregadas.")
N, p = X.shape

# ----------------------------------------------------------
# SIMULAÇÃO DE MONTE CARLO
# (Focado apenas em Acurácia)
# ----------------------------------------------------------

metricas_acuracia = []
# Outras listas de métricas removidas
resultados = []
R = 1 # (Preservado)

print(f"\nIniciando simulação de Monte Carlo com {R} rodadas...")
for r in range(R):
    
    idx = np.random.permutation(N)
    Xr = X[idx,:]
    yr = y[idx, :] # 'yr' é one-hot encoded, shape (N, 20)
    
    # Particionamento (80% treino, 20% teste)
    split_idx = int(N * 0.8)
    X_treino = Xr[:split_idx, :]
    y_treino = yr[:split_idx, :] # Shape (N_treino, 20)
    
    X_teste = Xr[split_idx:, :] 
    y_teste = yr[split_idx:, :] # Shape (N_teste, 20)
    
    # NORMALIZAÇÃO DOS DADOS [-1, 1] (Preservado)
    min_treino = X_treino.min(axis=0)
    max_treino = X_treino.max(axis=0)
    denominador = (max_treino - min_treino) + 1e-8
    X_treino_norm = 2 * (X_treino - min_treino) / denominador - 1
    X_teste_norm = 2 * (X_teste - min_treino) / denominador - 1

    # FIM DA NORMALIZAÇÃO
    
    # Topologia da rede (Preservada)
    # [Entrada (100), Oculta (30), Saída (20)]
    layer_dims = [X_treino.shape[1], 30, 20] 
    
    # y_treino.T tem shape (20, N_treino), que é o esperado pela MLP
    ps = MultilayerPerceptron(topology=layer_dims, X_train=X_treino_norm.T, Y_train=y_treino.T, max_epoch=100, learning_rate=0.01)
    ps.fit()
      
    # y_pred terá shape (20, N_teste)
    y_pred = ps.predict(X_teste_norm.T)
    
    # --- CÁLCULO DE ACURÁCIA MULTI-CLASSE ---
    
    # 1. Converter predições (softmax/logits) para a classe (índice) vencedora
    # axis=0 pois o shape é (20, N_teste)
    y_pred_classes = np.argmax(y_pred, axis=0) 
    
    # 2. Converter y_teste (one-hot) de volta para classes (índices)
    # axis=1 pois o shape é (N_teste, 20)
    y_true_classes = np.argmax(y_teste, axis=1)

    # 3. Calcular acurácia
    acc = np.mean(y_true_classes == y_pred_classes)
    
    # 4. Salvar resultados
    metricas_acuracia.append(acc)
    
    resultados.append({
        "acc": acc,
        "y_true": y_true_classes, # Salva classes (0-19)
        "y_pred": y_pred_classes, # Salva classes (0-19)
        "errors": ps.errors_per_epoch # Salva histórico de erro
    })
    
    if (r + 1) % 50 == 0 or (r + 1) == R:
        print(f"Rodada {r + 1}/{R} concluída.")

# ----------------------------------------------------------
# ----- PLOTAGEM (Acurácia, Matriz Confusão e Curvas) -----
# (Bloco de plotagem totalmente substituído)
# ----------------------------------------------------------

# 1. Encontrar melhor e pior rodada (baseado na acurácia)
if not metricas_acuracia:
    print("Nenhuma rodada foi executada. Encerrando.")
    sys.exit()

accs = np.array(metricas_acuracia)
best_idx = np.argmax(accs)
worst_idx = np.argmin(accs)

melhor = resultados[best_idx]
pior = resultados[worst_idx]

# Cores
GREENS = ['#006045', '#009966', '#00d492', '#a4f4cf']
REDS = ['#a50036', '#ec003f', '#ff637e', '#ffccd3']

# ----------------------------------------------------------
# A. PLOTAR MATRIZES DE CONFUSÃO (20x20)
# ----------------------------------------------------------

# Labels para os eixos (nomes das pastas)
labels_plot = unique_labels 

# Preparar dados
y_true_melhor = melhor["y_true"]
y_pred_melhor = melhor["y_pred"]
mc_melhor = Matriz_Confusao.conf_matriz(y_true_melhor, y_pred_melhor)

y_true_pior = pior["y_true"]
y_pred_pior = pior["y_pred"]
mc_pior = Matriz_Confusao.conf_matriz(y_true_pior, y_pred_pior)

# Plotar
fig_cm, (ax_cm_melhor, ax_cm_pior) = plt.subplots(1, 2, figsize=(28, 12)) # Figura maior

sns.heatmap(mc_melhor, annot=True, fmt='d', cmap="Greens", cbar=True,
            ax=ax_cm_melhor, xticklabels=labels_plot, yticklabels=labels_plot)
ax_cm_melhor.set_title(f'Melhor Acurácia - Matriz de Confusão (Rodada {best_idx+1})')
ax_cm_melhor.set_xlabel('Predito (Previsto)')
ax_cm_melhor.set_ylabel('Verdadeiro (Real)')
ax_cm_melhor.set_xticklabels(ax_cm_melhor.get_xticklabels(), rotation=90)
ax_cm_melhor.set_yticklabels(ax_cm_melhor.get_yticklabels(), rotation=0)

sns.heatmap(mc_pior, annot=True, fmt='d', cmap="Reds", cbar=True,
            ax=ax_cm_pior, xticklabels=labels_plot, yticklabels=labels_plot)
ax_cm_pior.set_title(f'Pior Acurácia - Matriz de Confusão (Rodada {worst_idx+1})')
ax_cm_pior.set_xlabel('Predito (Previsto)')
ax_cm_pior.set_ylabel('Verdadeiro (Real)')
ax_cm_pior.set_xticklabels(ax_cm_pior.get_xticklabels(), rotation=90)
ax_cm_pior.set_yticklabels(ax_cm_pior.get_yticklabels(), rotation=0)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# B. PLOTAR CURVAS DE APRENDIZADO (Função fornecida)
# ----------------------------------------------------------

# Para fazer a ponte entre a ESTRUTURA DE DADOS do 'main' e a ESTRUTURA
# ESPERADA pela função 'plotar_curva_aprendizado', criamos esta classe auxiliar.
class DummyModel:
    def __init__(self, errors):
        self.errors_per_epoch = errors

# ----------------------------------------------------------
# FUNÇÃO AUXILIAR (Exatamente como fornecida)
# ----------------------------------------------------------
def plotar_curva_aprendizado(resultados_rodada, n_classes, titulo_grafico, cor_plot):

        all_errors = []
        max_len = 0
        
        try:
            # Itera sobre os "modelos" (no nosso caso, haverá apenas 1)
            for model in resultados_rodada["models"]:
                
                history = model.errors_per_epoch 
                # ---------------------------------------------------

                if not hasattr(model, 'errors_per_epoch'):
                        print(f"O NOME DO MODELO É PSPS PARA OS ÍNTIMOS 🦄.")
                        raise AttributeError("Atributo 'errors_per_epoch' não encontrado no modelo.")

                all_errors.append(history)
                if len(history) > max_len:
                    max_len = len(history)
                    
        except AttributeError as e:
            print(str(e))
            print(f"Não foi possível gerar a curva de aprendizado para: {titulo_grafico}.\n")
            return 
        
        
        if max_len == 0:
            print(f"Nenhum histórico de erro encontrado para: {titulo_grafico}.")
            return

        # 'n_classes' aqui é usado para 'padded_errors', mas na verdade
        # deveria ser o n_modelos (len(all_errors)).
        # Vamos usar 'len(all_errors)' que é mais robusto.
        n_models_in_rodada = len(all_errors)
        if n_models_in_rodada == 0:
             print(f"Nenhum modelo encontrado para: {titulo_grafico}.")
             return
             
        padded_errors = np.zeros((n_models_in_rodada, max_len))
        
        for i, history in enumerate(all_errors):
            last_error = history[-1] if len(history) > 0 else 0
            padded_errors[i, :len(history)] = history
            padded_errors[i, len(history):] = last_error 
    
        # Calcular média e desvio padrão (com 1 modelo, std será 0)
        mean_errors = np.mean(padded_errors, axis=0)
        std_errors = np.std(padded_errors, axis=0)
    
        epochs = np.arange(1, max_len + 1)
    
        # Plotar o gráfico
        plt.figure(figsize=(12, 7))
        plt.plot(epochs, mean_errors, color=cor_plot, lw=2, label='Erro Quadrático Médio (MSE) - Média')
        
        # Plotar a banda de desvio padrão
        plt.fill_between(epochs, 
                         mean_errors - std_errors, 
                         mean_errors + std_errors, 
                         color=cor_plot, 
                         alpha=0.2, 
                         label='Desvio Padrão (± 1 std)')
        
        plt.title(f'Curva de Aprendizado Média ({titulo_grafico})', fontsize=16, fontweight='bold')
        plt.xlabel('Época', fontsize=12)
        plt.ylabel('Erro Quadrático Médio (MSE) - Escala Log', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Usar escala logarítmica no eixo Y é bom para ver a convergência
        plt.yscale('log')
        
        plt.tight_layout()
        plt.show()

# ----------------------------------------------------------
# CHAMADAS PARA AS CURVAS DE APRENDIZADO
# ----------------------------------------------------------

# 1. Empacotar dados do "melhor" para a função
best_for_plot = {
    "models": [DummyModel(melhor.get("errors", []))]
}
# 2. Empacotar dados do "pior" para a função
worst_for_plot = {
    "models": [DummyModel(pior.get("errors", []))]
}

# Plotar a curva da melhor rodada 
plotar_curva_aprendizado(
    best_for_plot, 
    n_classes, # Passando n_classes (20)
    f"Melhor Rodada - N° {best_idx+1}", 
    GREENS[1] 
)

# Plotar a curva da pior rodada 
plotar_curva_aprendizado(
    worst_for_plot, 
    n_classes, 
    f"Pior Rodada - N° {worst_idx+1}", 
    REDS[1] 
)


# ----------------------------------------------------------
# --- Finalização (Apenas Acurácia) ---
# ----------------------------------------------------------
print("\nSimulação concluída.")
print("Estatísticas gerais (todas as rodadas):")
Avaliador.print_stat("Acurácia", metricas_acuracia)
# Outras impressões removidas

plt.show() # Garante que todos os plots abertos sejam exibidos
