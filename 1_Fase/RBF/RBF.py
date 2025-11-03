import numpy as np
import matplotlib.pyplot as plt

class RBF:

    def __init__(self, X_train, y_train, num_centers=10, sigma=1.0, learning_rate=1e-2, max_epoch=100):
        
        self.p, self.N = X_train.shape 
        self.X_train = X_train
        self.d = y_train.flatten()      
        
        self.q = num_centers           
        self.sigma = sigma
        self.lr = learning_rate
        self.max_epoch = max_epoch
        
        self.errors_per_epoch = []

    
        idx = np.random.permutation(self.N)
        self.centers = self.X_train[:, idx[:self.q]] 
        self.w = np.random.random_sample((self.q + 1, 1)) - 0.5


    def _gaussian(self, x, center):
        distance_sq = np.linalg.norm(x - center)**2
        
        # yi(t) = exp(...)
        return np.exp(-distance_sq / (2 * self.sigma**2))

    
    def _transform(self, X):
     
        p_X, N_X = X.shape
        
        G = np.zeros((self.q, N_X))
        
        for i in range(N_X):  
            for j in range(self.q): 
                
                x_i = X[:, i].reshape(-1, 1)
                c_j = self.centers[:, j].reshape(-1, 1)
                
                
                G[j, i] = self._gaussian(x_i, c_j)
        
        
        G_bias = np.vstack((-np.ones((1, N_X)), G))
        
        return G_bias

    
    def EQM(self, G_bias, d_true):
        
        N_samples = d_true.shape[0]
       
        u = self.w.T @ G_bias
        e = d_true.reshape(1, -1) - u
        
        eqm = np.sum(e**2) / (2 * N_samples)
        return eqm

        
    def fit(self):
        G_bias = self._transform(self.X_train)
        
        
        self.errors_per_epoch.append(self.EQM(G_bias, self.d))
        
        epochs = 0
        while epochs < self.max_epoch:
            
            for k in range(self.N):
                g_k = G_bias[:, k].reshape(self.q + 1, 1)
                u_k = (self.w.T @ g_k)[0, 0]
                
                d_k = self.d[k] 
                e_k = d_k - u_k 
                self.w = self.w + self.lr * e_k * g_k
            
            # EQM da época
            self.errors_per_epoch.append(self.EQM(G_bias, self.d))
            epochs += 1


    def activation_function(self, u):
        """ 
        Função de ativação
        """
        return 1 if u >= 0 else -1

        
    def predict(self, X_test):
        
        G_test_bias = self._transform(X_test)
        u_test = self.w.T @ G_test_bias
        
        # Aplica a função de ativação para classificar
        y_pred = [self.activation_function(u) for u in u_test.flatten()]
            
        return np.array(y_pred).reshape(-1, 1)