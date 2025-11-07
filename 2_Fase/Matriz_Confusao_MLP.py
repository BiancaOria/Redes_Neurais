import numpy as np

class Matriz_Confusao:
    @staticmethod
    def conf_matriz(y_true, y_pred):
        
        y_true = np.where(np.array(y_true) >= 0, 1, -1)
        y_pred = np.where(np.array(y_pred) >= 0, 1, -1)

        # Inicializa contadores
        count_VP = 0  # Verdadeiro Positivo
        count_VN = 0  # Verdadeiro Negativo
        count_FP = 0  # Falso Positivo
        count_FN = 0  # Falso Negativo

        # Percorre as listas
        for i in range(len(y_true)):
            if y_true[i] == 1 and y_pred[i] == 1:
                count_VP += 1
            elif y_true[i] == -1 and y_pred[i] == -1:
                count_VN += 1
            elif y_true[i] == -1 and y_pred[i] == 1:
                count_FP += 1
            elif y_true[i] == 1 and y_pred[i] == -1:
                count_FN += 1

        # Cria matriz de confusão 2x2
        # Ordem: [ [VP, FN],
        #          [FP, VN] ]
        mc = np.array([
            [count_VP, count_FN],
            [count_FP, count_VN]
        ])

        # Retorna tudo
        return mc, count_VP, count_VN, count_FP, count_FN
