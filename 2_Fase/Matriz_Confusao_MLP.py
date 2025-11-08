import numpy as np

class Matriz_Confusao:
    """
    Gera uma matriz de confusão N x N para classificação multiclasse.
    """
    @staticmethod
    def conf_matriz(y_true, y_pred, labels=None):
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Verifica tamanhos
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError(f"Tamanhos incompatíveis: y_true ({y_true.shape}) e y_pred ({y_pred.shape})")

        # Descobre labels se não forem fornecidos
        if labels is None:
            labels = sorted(list(set(y_true) | set(y_pred)))

        n_labels = len(labels)

        # Inicializa matriz N×N
        mc = np.zeros((n_labels, n_labels), dtype=int)

        # Mapa label → índice (para acesso rápido)
        label_to_index = {label: idx for idx, label in enumerate(labels)}

        # Percorre amostras e contabiliza
        for yt, yp in zip(y_true, y_pred):
            if yt in label_to_index and yp in label_to_index:
                i = label_to_index[yt]  # índice do verdadeiro
                j = label_to_index[yp]  # índice do predito
                mc[i, j] += 1
            # caso contrário, ignora (pode ocorrer se houver valores fora da faixa)

        return mc, labels
