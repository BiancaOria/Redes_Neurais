import numpy as np
import matplotlib.pyplot as plt

class RBF:
    """
    Implementação da Rede de Função de Base Radial (RBF) para classificação.
    
    A arquitetura é composta por uma camada de entrada, uma camada oculta RBF
    e uma camada de saída linear.
    
    O treinamento segue duas etapas:
    1. Definição dos centros (aqui por seleção aleatória).
    2. Treinamento dos pesos da camada de saída via Regra Delta (LMS).
    """

    def __init__(self, X_train, y_train, num_centers=10, sigma=1.0, learning_rate=1e-2, max_epoch=100):
        """
        Inicializa a rede RBF.
        
        Parâmetros:
        -----------
        X_train : numpy.ndarray
            Dados de treino (formato [features, amostras]), seguindo o estilo ADALINE.
        y_train : numpy.ndarray
            Rótulos de treino (formato [amostras, 1] ou [amostras,])
        num_centers : int
            O número 'q' de neurônios (centros) na camada oculta RBF.
        sigma : float
            O "raio" (largura) da função Gaussiana.
        learning_rate : float
            Taxa de aprendizado (eta) para a Regra Delta da camada de saída.
        max_epoch : int
            Número de épocas para o treinamento da camada de saída.
        """
        
        self.p, self.N = X_train.shape  # p = features, N = amostras
        self.X_train = X_train
        self.d = y_train.flatten()      # Saídas desejadas
        
        self.q = num_centers            # Número de centros (neurônios ocultos)
        self.sigma = sigma
        self.lr = learning_rate
        self.max_epoch = max_epoch
        
        self.errors_per_epoch = []

        # --- Etapa 1: Definir Centros (Não supervisionado) ---
        # Usamos o método de "Seleção aleatória de centros".
        # Seleciona 'q' amostras aleatórias de X_train para serem os centros.
        idx = np.random.permutation(self.N)
        self.centers = self.X_train[:, idx[:self.q]] # Formato [p, q]
        
        # --- Etapa 2: Inicializar Pesos da Camada de Saída (w) ---
        # A camada de saída recebe 'q' entradas da camada RBF + 1 bias.
        # Usamos a mesma inicialização aleatória do ADALINE.
        self.w = np.random.random_sample((self.q + 1, 1)) - 0.5


    def _gaussian(self, x, center):
        """ 
        Função de ativação RBF (Gaussiana).
        Calcula a saída de um neurônio oculto.
        
        ui(t) = ||x(t) - ci(t)||
        yi(t) = exp(-ui(t)^2 / (2 * sigma_i^2))
        """
        # ui(t) = ||x(t) - ci(t)||
        distance_sq = np.linalg.norm(x - center)**2
        
        # yi(t) = exp(...)
        return np.exp(-distance_sq / (2 * self.sigma**2))

    
    def _transform(self, X):
        """ 
        Transforma os dados de entrada X na saída da camada RBF (G).
        Esta é a matriz 'Z' descrita na seção de Mínimos Quadrados.
        
        X tem formato [p, N_samples]
        Retorna G_bias com formato [q+1, N_samples]
        """
        p_X, N_X = X.shape
        
        # G é a matriz de design (saída da camada RBF)
        G = np.zeros((self.q, N_X))
        
        for i in range(N_X):  # Para cada amostra
            for j in range(self.q): # Para cada centro
                # Pega a amostra i e o centro j
                x_i = X[:, i].reshape(-1, 1)
                c_j = self.centers[:, j].reshape(-1, 1)
                
                # Calcula a saída gaussiana
                G[j, i] = self._gaussian(x_i, c_j)
        
        # Adiciona a entrada de bias (-1), igual ao ADALINE
        # A referência usa +1, mas seguimos o -1 do ADALINE
        G_bias = np.vstack((-np.ones((1, N_X)), G))
        
        return G_bias

    
    def EQM(self, G_bias, d_true):
        """ 
        Calcula o Erro Quadrático Médio na CAMADA DE SAÍDA.
        A fórmula é a mesma do ADALINE, mas 'u(t)' é
        calculado a partir das saídas da RBF (G_bias), não de X.
        """
        N_samples = d_true.shape[0]
        
        # u é a saída linear da camada final: u(t) = w^T * z(t)
        u = self.w.T @ G_bias
        
        # e = d - u
        e = d_true.reshape(1, -1) - u
        
        # EQM = 1/(2N) * sum(e^2)
        eqm = np.sum(e**2) / (2 * N_samples)
        return eqm

        
    def fit(self):
        """ 
        Treina os pesos (w) da CAMADA DE SAÍDA usando a 
        Regra Delta (LMS), como no ADALINE.
        """
        
        # 1. Transforma todos os dados de treino UMA VEZ.
        #    A camada RBF (centros e sigmas) não muda durante o fit.
        G_bias = self._transform(self.X_train) # Formato [q+1, N]
        
        # Adiciona o erro inicial
        self.errors_per_epoch.append(self.EQM(G_bias, self.d))
        
        epochs = 0
        while epochs < self.max_epoch:
            
            # Loop de amostra por amostra (LMS)
            for k in range(self.N):
                # g_k é a saída da camada RBF (já com bias) para a amostra k
                # (Corresponde a z(t) em)
                g_k = G_bias[:, k].reshape(self.q + 1, 1)
                
                # u_k é a saída LINEAR final (w^T * g_k)
                u_k = (self.w.T @ g_k)[0, 0]
                
                d_k = self.d[k] # Saída desejada
                
                # e_k é o erro LINEAR (d_k - u_k), como no ADALINE
                e_k = d_k - u_k 
                
                # Atualização de pesos (Regra Delta do ADALINE)
                # w(t+1) = w(t) + eta * e(t) * x(t)
                # Aqui, 'x(t)' é a entrada desta camada, que é 'g_k'
                self.w = self.w + self.lr * e_k * g_k
            
            # Calcula o EQM da época
            self.errors_per_epoch.append(self.EQM(G_bias, self.d))
            epochs += 1


    def activation_function(self, u):
        """ 
        Função de ativação de SAÍDA (classificação).
        Para classificação binária, usamos a função degrau (sinal).
        Isso corresponde à fase de teste.
        """
        return 1 if u >= 0 else -1

        
    def predict(self, X_test):
        """
        Prevê os rótulos para novos dados X_test.
        X_test tem formato [features, N_test_samples]
        """
        
        # 1. Transforma os dados de teste usando as RBFs
        G_test_bias = self._transform(X_test)
        
        # 2. Calcula a saída linear (u = w^T * z)
        u_test = self.w.T @ G_test_bias
        
        # 3. Aplica a função de ativação para classificar
        y_pred = [self.activation_function(u) for u in u_test.flatten()]
            
        return np.array(y_pred).reshape(-1, 1)