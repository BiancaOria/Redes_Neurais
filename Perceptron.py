import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self,X_train,y_train,learning_rate=1e-3,max_epoch=500,plot=True):
        self.p, self.N = X_train.shape
        self.X_train = np.vstack((
            -np.ones((1,self.N)), X_train
        ))
        self.max_epoch = max_epoch
        self.d = np.array(y_train).flatten()
        self.COLORS = ['#f6cfff', "#677f8b", '#c4b4ff', '#fccee8', '#96f7e4', '#a4f4cf', '#fef3c6', '#ffccd3']
        self.lr = learning_rate
        self.w = np.zeros((self.p+1,1))
        self.w = np.random.random_sample((self.p+1,1))-.5 # arbritario ?        
        self.plot = plot
        self.i = 0
        if plot:
            plt.ion()
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
            self.ax.set_title("Visualização Inicial dos Dados (Espiral 3D)")
            
            #self.draw_line()
        
        
    def draw_line(self,color,alpha):
        
        # cria grid 2D
        x1 = np.linspace(self.X_train[1].min()-1, self.X_train[1].max()+1, 10)
        x2 = np.linspace(self.X_train[2].min()-1, self.X_train[2].max()+1, 10)
        X1, X2 = np.meshgrid(x1, x2)
        # plano 3D do Perceptron
        # Z = np.zeros_like(X1)
        Z = (self.w[0]*(-1) + self.w[1]*X1 + self.w[2]*X2)# tem o menos no w0 ?
        # desenha no subplot 3D
        self.ax.plot_surface(X1, X2, Z, color=color, alpha=alpha)  
        
        
    def activation_function(self, u):
        return 1 if u>=0 else -1
    
    def fit(self):
        epochs = 0 #precisa?
        error = True
        while error and epochs < self.max_epoch:
            error = False
            for k in range(self.N):
                x_k = self.X_train[:,k].reshape(self.p+1,1)
                u_k = (self.w.T@x_k)[0,0]
                y_k = self.activation_function(u_k)
                d_k = self.d[k]
                e_k = d_k - y_k
                self.w = self.w + self.lr*e_k*x_k
                if e_k!=0:
                    error = True
            epochs+=1 #precisa?   
            # plt.pause(.4)
            # self.draw_line(color='b',alpha=.01)

        self.draw_line(color='y',alpha=.5)    
        # plt.pause(.4)
        # if self.plot:
        #     self.draw_line(color='y',alpha=.1)
        #     plt.show()
        #     plt.show(block=True)
    def predict(self, X_test):
            
        p_test, N_test = X_test.shape
        
        X_test_bias = np.vstack((
            -np.ones((1, N_test)), X_test
        ))
        
        u_test = self.w.T @ X_test_bias
        
        y_pred = [self.activation_function(u) for u in u_test.flatten()]
            
        return np.array(y_pred).reshape(-1, 1)