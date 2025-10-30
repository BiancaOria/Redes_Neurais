import numpy as np

class Matriz_Confusao:
    def conf_matriz(y_true, y_pred, labels=[1, -1]):
        
        cm = np.zeros((2, 2), dtype=int)
        
        
        label_to_index = {label: idx for idx, label in enumerate(labels)}
        
        for yt, yp in zip(y_true, y_pred):
            i = label_to_index[yt]
            j = label_to_index[yp]
            cm[i, j] += 1
        
        return cm