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
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Número total de amostras
        total = len(y_true)
        # Número de acertos
        acertos = np.sum(y_true == y_pred)
        # Acurácia
        acc = acertos / total if total > 0 else 0.0
        
        return acc
    
    @staticmethod 
    def get_pior(metric_name,resultados):
        valores = [r[metric_name] for r in resultados]
        idx_worst = np.argmin(valores)
        return resultados[idx_worst]  
    def get_melhor(metric_name,resultados):
        valores = [r[metric_name] for r in resultados]
        idx_best = np.argmax(valores)
        
        return resultados[idx_best]  