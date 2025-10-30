import numpy as np

class Avaliador:
    
    @staticmethod  
    def print_stat(nome_metrica, valores):
        media = np.mean(valores)
        std = np.std(valores)
        maximo = np.max(valores)
        minimo = np.min(valores)
        print(f"{nome_metrica:15}: Média = {media:.4f}  |  Desvio Padrão = {std:.4f} | Máxima = {maximo:.4f}   | Mínima = {minimo:.4f}   |")
    
    
    @staticmethod 
    def calcular_metricas(y_true, y_pred): 
        
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
    
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == -1) & (y_pred == -1))
        FP = np.sum((y_true == -1) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == -1))
        
        total = TP + TN + FP + FN
        
        # Acurácia
        acuracia = (TP + TN) / total if total > 0 else 0
        
        # Sensibilidade (Recall)
        denominador_sens = (TP + FN)
        sensibilidade = TP / denominador_sens if denominador_sens > 0 else 0
        
        # Especificidade
        denominador_espec = (TN + FP)
        especificidade = TN / denominador_espec if denominador_espec > 0 else 0
        
        # Precisão
        denominador_prec = (TP + FP)
        precisao = TP / denominador_prec if denominador_prec > 0 else 0 
        
        # F1-Score
        denominador_f1 = (precisao + sensibilidade)
        f1_score = 2 * (precisao * sensibilidade) / denominador_f1 if denominador_f1 > 0 else 0
        
        return acuracia, sensibilidade, especificidade, precisao, f1_score
    
    @staticmethod 
    def get_pior(metric_name,resultados):
        valores = [r[metric_name] for r in resultados]
        idx_worst = np.argmin(valores)
        return resultados[idx_worst]  
    def get_melhor(metric_name,resultados):
        valores = [r[metric_name] for r in resultados]
        idx_best = np.argmax(valores)
        
        return resultados[idx_best]  