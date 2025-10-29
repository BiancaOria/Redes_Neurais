import numpy as np
import matplotlib.pyplot as plt

class ADALINE:
    def __init__(self,X_train,y_train,learning_rate=1e-3,max_epoch=10000,tol=1e-5,plot=True):
        self.p, self.N = X_train.shape
        self.X_train = np.vstack((
            -np.ones((1,self.N)), X_train
        ))
        self.max_epoch = max_epoch
        self.tol = tol
        self.d = y_train
        self.lr = learning_rate
        self.w = np.zeros((self.p+1,1))
        self.w = np.random.random_sample((self.p+1,1))-.5
        self.plot = plot
        if plot:
            self.fig = plt.figure(1)
            self.ax = self.fig.add_subplot(projection='3d')
            self.ax.scatter(self.X_train[1, self.d==1],
                            self.X_train[2, self.d==1],
                            self.d[self.d==1],
                            c='r', marker='s', s=120, edgecolor='k')
            
            self.ax.scatter(self.X_train[1, self.d==-1],
                            self.X_train[2, self.d==-1],
                            self.d[self.d==-1],
                            c='b', marker='o', s=120, edgecolor='k')
            margin = 1  # margem extra
            x_min, x_max = self.X_train[1].min() - margin, self.X_train[1].max() + margin
            y_min, y_max = self.X_train[2].min() - margin, self.X_train[2].max() + margin
            z_min, z_max = self.d.min() - margin, self.d.max() + margin

            self.ax.set_xlim(x_min, x_max)
            self.ax.set_ylim(y_min, y_max)
            self.ax.set_zlim(z_min, z_max)
            
            self.ax.set_xlabel("Variável 1")
            self.ax.set_ylabel("Variável 2")
            self.ax.set_zlabel("Resultado")
            self.ax.set_title("Gráfico 3D do conjunto de dados")
            
            self.draw_line()
        
    def draw_line(self,c='k',alpha=1,lw=2):
        x1 = np.linspace(self.X_train[1].min()-1, self.X_train[1].max()+1, 10)
        x2 = np.linspace(self.X_train[2].min()-1, self.X_train[2].max()+1, 10)
        X1, X2 = np.meshgrid(x1, x2)
        
        # plano 3D do Perceptron
        Z = np.zeros_like(X1)
        
        # desenha no subplot 3D
        self.ax.plot_surface(X1, X2, Z, color=color, alpha=alpha)
        plt.show()    
        
    def activation_function(self, u):
        return 1 if u>=0 else -1
    
    def EQM(self):
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
        hist_eqm = []
        while epochs < self.max_epoch and abs(EQM1 - EQM2)>self.tol:
            EQM1 = self.EQM()
            hist_eqm.append(EQM1)
            for k in range(self.N):
                x_k = self.X_train[:,k].reshape(self.p+1,1)
                u_k = (self.w.T@x_k)[0,0]
                d_k = self.d[k]
                e_k = d_k-u_k
                self.w = self.w + self.lr*e_k*x_k
            EQM2 = self.EQM()
            plt.pause(.1)
            self.draw_line(c='r',alpha=.5)
            epochs+=1
        hist_eqm.append(EQM2)
        plt.pause(.1)
        self.draw_line(c='g',alpha=1,lw=4)
        
#         plt.figure(2)
#         plt.plot(hist_eqm)
#         plt.grid()
#         plt.title("Curva de Aprendizagem (ADALINE)")
#         plt.show()
        
        


