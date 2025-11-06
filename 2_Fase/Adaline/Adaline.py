import numpy as np
import matplotlib.pyplot as plt

class ADALINE:
    def __init__(self,X_train,y_train,learning_rate=1e-3,max_epoch=1000,tol=1e-12,plot=True):
        self.p, self.N = X_train.shape
        self.X_train = np.vstack((
            -np.ones((1,self.N)), X_train
        ))
        self.max_epoch = max_epoch
        self.tol = tol
        self.d = y_train.flatten()
        self.lr = learning_rate
        self.w = np.zeros((self.p+1,1))
        self.w = np.random.random_sample((self.p+1,1))-.5
        self.plot = plot
        self.x1 = 0
        self.errors_per_epoch = []
        self.w_per_epoch = [] 
        if plot:
            # plt.ion() # ?
            # self.fig = plt.figure(2)
            # self.ax = self.fig.add_subplot()
            # self.ax.scatter(self.X_train[1,self.d[:]==1],
            #                 self.X_train[2,self.d[:]==1],c='#ff2cc9', marker='s', s=120, edgecolor='k')
            # self.ax.scatter(self.X_train[1,self.d[:]==-1],
            #                 self.X_train[2,self.d[:]==-1],c='#8e51ff', marker='o', s=120, edgecolor='k')
            margin = 0  # margem extra
            x_min, x_max = self.X_train[1].min() - margin, self.X_train[1].max() + margin
            y_min, y_max = self.X_train[2].min() - margin, self.X_train[2].max() + margin
            # self.ax.grid(True)

            # self.ax.set_xlim(x_min, x_max)
            # self.ax.set_ylim(y_min, y_max)
            
            
            self.x1 = np.linspace(x_min,x_max)
            
            # self.ax.set_xlabel("Variável 1")
            # self.ax.set_ylabel("Variável 2")
    
            # self.ax.set_title("Plano de Regressão do ADALINE")
            # self.ax.legend()
            
            
        

        
    def activation_function(self, u):
        return 1 if u >= 0 else - 1
        
    #classe nova pra a segundfa fase
    def predict_raw(self, X_test):
        p_test, N_test = X_test.shape
        X_test_bias = np.vstack((
            -np.ones((1, N_test)), X_test
        ))
        u_test = self.w.T @ X_test_bias  # produto escalar (1, N_test)
        return u_test.flatten()
    
    def EQM(self):#conferido pelo psudocodigo
        eqm = 0
        for k in range(self.N):
            x_k = self.X_train[:,k].reshape(self.p+1,1)
            u_k = (self.w.T@x_k)[0,0]
            d_k = self.d[k]
            eqm += (d_k-u_k)**2
        return eqm/(2*self.N)
    def fit(self):
        epochs = 0
        EQM1 = 0
        EQM2 = 1
        while epochs < self.max_epoch and abs(EQM1 - EQM2) > self.tol:
            EQM1 = self.EQM()
            self.errors_per_epoch.append(EQM1)
            
            for k in range(self.N):
                x_k = self.X_train[:,k].reshape(self.p+1,1)
                u_k = (self.w.T@x_k)[0,0] #produto escalar
                d_k = self.d[k]#saida desejada
                e_k = d_k-u_k #erro quadratico
                self.w  = self.w + self.lr*e_k*x_k #lr taxa de aprendizagem
            self.w_per_epoch.append(self.w.flatten())
            epochs+=1
            EQM2 = self.EQM()#até aqui ok, conferigo pelo pseudo codigo
            # plt.pause(.1)
        self.errors_per_epoch.append(EQM2)    
        
        
    def predict(self, X_test):
            
        p_test, N_test = X_test.shape
        
        X_test_bias = np.vstack((
            -np.ones((1, N_test)), X_test
        ))
        
        u_test = self.w.T @ X_test_bias
        
        y_pred = [self.activation_function(u) for u in u_test.flatten()]
            
        return np.array(y_pred).reshape(-1, 1)
        
        


