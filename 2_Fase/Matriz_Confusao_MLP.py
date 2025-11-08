import numpy as np

class Matriz_Confusao:
    """
    Matriz de confusão N x N e versão binária 2x2 (TP, FN, FP, TN) usando lógica one-vs-all.
    """

    @staticmethod
    def conf_matriz(y_true, y_pred, labels=None, binarizar=False):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError(f"Tamanhos incompatíveis: y_true ({y_true.shape}) e y_pred ({y_pred.shape})")

        # Matriz N×N clássica
        if not binarizar:
            if labels is None:
                labels = sorted(list(set(y_true) | set(y_pred)))
            n_labels = len(labels)
            mc = np.zeros((n_labels, n_labels), dtype=int)
            label_to_index = {label: idx for idx, label in enumerate(labels)}

            for yt, yp in zip(y_true, y_pred):
                i = label_to_index[yt]
                j = label_to_index[yp]
                mc[i, j] += 1
            return mc, labels

        # Matriz 2x2 binária agregando multiclasses (one-vs-all)
        if labels is None:
            labels = sorted(list(set(y_true) | set(y_pred)))

        TP_total = 0
        FN_total = 0
        FP_total = 0
        TN_total = 0

        for label in labels:
            y_true_bin = np.where(y_true == label, 1, -1)
            y_pred_bin = np.where(y_pred == label, 1, -1)

            TP = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
            FN = np.sum((y_true_bin == 1) & (y_pred_bin == -1))
            FP = np.sum((y_true_bin == -1) & (y_pred_bin == 1))
            TN = np.sum((y_true_bin == -1) & (y_pred_bin == -1))

            TP_total += TP
            FN_total += FN
            FP_total += FP
            TN_total += TN

        mc_bin = np.array([[TP_total, FN_total],
                           [FP_total, TN_total]], dtype=int)
        return mc_bin, ['Positivo', 'Negativo']
