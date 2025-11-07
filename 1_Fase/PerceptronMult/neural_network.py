import numpy as np
import matplotlib.pyplot as plt

class MultilayerPerceptron:
    def __init__(self,X_train:np.ndarray, Y_train:np.ndarray, topology:list, learning_rate = 1e-3, plot=True, tol = 1e-12, max_epoch = 10000):
        '''
        X_train (p x N)
        Y_train (C x N) ou (1 x N)
        '''
        
        self.p, self.N = X_train.shape
        self.m = Y_train.shape[0]
        self.X_train = np.vstack((
            -np.ones((1,self.N)), X_train
        ))
        self.D = Y_train
        
        self.tol = tol
        self.lr = learning_rate
        self.errors_per_epoch = []
        
        topology.append(self.m)
        print(topology)
        self.W = [None]*len(topology)
        Z = 0
        for i in range(len(self.W)):
            if i == 0:
                W = np.random.random_sample((topology[i],self.p+1))-.5
            else:
                W = np.random.random_sample((topology[i], topology[i-1]+1))-.5
            self.W[i] = W
            Z += W.size
        # print(f"Rede MLP com {Z} parâmetros")
        self.y = [None]*len(topology)
        self.u = [None]*len(topology)
        self.delta = [None]*len(topology)
        self.max_epoch = max_epoch

        if plot:
            self.fig = plt.figure(2)
            self.ax = self.fig.add_subplot()
            self.ax.scatter(self.X_train[1, self.D.T[:, 0] == 1],
                            self.X_train[2, self.D.T[:, 0] == 1],c='#ff2cc9', marker='s', s=120, edgecolor='k')
            self.ax.scatter(self.X_train[1, self.D.T[:, 0] == -1],
                            self.X_train[2, self.D.T[:, 0] == -1],c='#8e51ff', marker='o', s=120, edgecolor='k')
            margin = 0
            x_min, x_max = self.X_train[1].min() - margin, self.X_train[1].max() + margin
            y_min, y_max = self.X_train[2].min() - margin, self.X_train[2].max() + margin

            self.ax.set_xlim(x_min, x_max)
            self.ax.set_ylim(y_min, y_max)
            self.x1 = np.linspace(x_min, x_max)
            self.ax.set_xlabel("Variável 1")
            self.ax.set_ylabel("Variável 2")
            self.ax.set_title("Plano de Regressão do Perceptron")
    
    def g(self,u):
        return (1-np.exp(-u))/(1+np.exp(-u))
    
    def g_d(self, u):
        y = self.g(u)
        return .5*(1 - y**2)
    
    def forward(self,x):
        for i,W in enumerate(self.W):
            if i == 0:
                self.u[i] = W@x                
            else:
                yb = np.vstack((
                    -np.ones((1,1)), self.y[i-1]
                ))
                self.u[i] = W@yb
            self.y[i] = self.g(self.u[i])
        return self.y
            
    # METODO DO CIRILLO
    """ def predict(self, x):
        self.forward(x)
        MC = np.zeros((self.m,self.m))
        return self.y[-1] """

    def predict(self, X_test):
        p_test, N_test = X_test.shape
        X_test_bias = np.vstack((-np.ones((1, N_test)), X_test))
        outputs = []
        for n in range(N_test):
            x = X_test_bias[:, n].reshape(p_test + 1, 1)
            y = self.forward(x)
            outputs.append(y[-1])
        return np.array(outputs)
    
    def EQM(self):
        s = 0
        for k in range(self.N):
            x_k = self.X_train[:,k].reshape(self.p+1,1)
            self.forward(x_k)
            y = self.y[-1]
            d = self.D[:,k].reshape(self.m,1)
            e = d - y
            s += np.sum(e**2)
            
        return s/(2*self.N)
            
    def backward(self,e,x):
        for i in range(len(self.W)-1,-1,-1):
            if i == len(self.W)-1:
                yb = np.vstack((
                   -1,
                    self.y[i-1]
                ))
                self.delta[i] = self.g_d(self.u[i]) * e
                self.W[i] = self.W[i] + self.lr*(self.delta[i]@yb.T)
            elif i == 0:
                Wnb = self.W[i+1][:,1:]
                self.delta[i] = self.g_d(self.u[i]) * (Wnb.T@self.delta[i+1])
                self.W[i] = self.W[i] + self.lr*(self.delta[i]@x.T)                
            else:
                yb = np.vstack((
                   -1,
                    self.y[i-1]
                ))
                Wnb = self.W[i+1][:,1:]
                self.delta[i] = self.g_d(self.u[i]) * (Wnb.T@self.delta[i+1])
                self.W[i] = self.W[i] + self.lr*(self.delta[i]@yb.T)
                
    
    def fit(self):
        epoch = 0
        EQM = self.EQM()
        # print(f'EQM: {EQM:.15f}, época: {epoch}')
        while epoch < self.max_epoch and EQM > self.tol:
            # t1 = time()
            for k in range(self.N):
                x_k = self.X_train[:,k].reshape(self.p+1,1)
                #Forward
                self.forward(x_k)
                y = self.y[-1]
                d = self.D[:,k].reshape(self.m,1)
                e = d - y
                #Backward
                self.backward(e,x_k)
            # t2 = time()
            EQM = self.EQM()
            self.errors_per_epoch.append(EQM)
            # print(f'EQM: {EQM:.15f}, época: {epoch}, Tempo: {t2-t1:.5f}s')            
            epoch+=1
        
































class Perceptron:
    def __init__(self,X_train,y_train,learning_rate=1e-3,plot=True):
        self.p, self.N = X_train.shape
        self.X_train = np.vstack((
            -np.ones((1,self.N)), X_train
        ))
        self.d = y_train
        self.lr = learning_rate
        self.w = np.zeros((self.p+1,1))
        self.w = np.random.random_sample((self.p+1,1))-.5
        self.plot = plot
        if plot:
            self.fig = plt.figure(1)
            self.ax = self.fig.add_subplot()
            self.ax.scatter(self.X_train[1,self.d[:]==1],
                            self.X_train[2,self.d[:]==1],marker='s',s=120)
            self.ax.scatter(self.X_train[1,self.d[:]==-1],
                            self.X_train[2,self.d[:]==-1],marker='o',s=120)
            self.ax.set_xlim(-1,7)
            self.ax.set_ylim(-1,7)
            self.x1 = np.linspace(-2,10)
            self.draw_line()
        
    def draw_line(self,c='k',alpha=1,lw=2):
        x2 = -self.w[1,0]/self.w[2,0]*self.x1 + self.w[0,0]/self.w[2,0]
        x2 = np.nan_to_num(x2)
        plt.plot(self.x1,x2,c=c,alpha=alpha,lw=lw)
        
    def activation_function(self, u):
        return 1 if u>=0 else -1
    
    def fit(self):
        epochs = 0
        error = True
        while error:
            error = False
            for k in range(self.N):
                x_k = self.X_train[:,k].reshape(self.p+1,1)
                u_k = (self.w.T@x_k)[0,0]
                y_k = self.activation_function(u_k)
                d_k = self.d[k]
                e_k = d_k - y_k
                if e_k!=0:
                    error = True
                self.w = self.w + self.lr*e_k*x_k
            
            plt.pause(.4)
            self.draw_line(c='r',alpha=.5)
            epochs+=1
        plt.pause(.4)
        self.draw_line(c='g',alpha=1,lw=4)
        plt.show()
class ADALINE:
    def __init__(self,X_train,y_train,learning_rate=1e-3,max_epoch=10000,tol=1e-5,plot=True):
        self.p, self.N = X_train.shape
        self.X_train = np.vstack((
            -np.ones((1,self.N)), X_train
        ))
        self.tol = tol
        self.max_epoch = max_epoch
        self.d = y_train
        self.lr = learning_rate
        self.w = np.zeros((self.p+1,1))
        self.w = np.random.random_sample((self.p+1,1))-.5
        self.plot = plot
        if plot:
            self.fig = plt.figure(1)
            self.ax = self.fig.add_subplot()
            self.ax.scatter(self.X_train[1,self.d[:]==1],
                            self.X_train[2,self.d[:]==1],marker='s',s=120)
            self.ax.scatter(self.X_train[1,self.d[:]==-1],
                            self.X_train[2,self.d[:]==-1],marker='o',s=120)
            self.ax.set_xlim(-1,7)
            self.ax.set_ylim(-1,7)
            self.x1 = np.linspace(-2,10)
            self.draw_line()
        
    def draw_line(self,c='k',alpha=1,lw=2):
        x2 = -self.w[1,0]/self.w[2,0]*self.x1 + self.w[0,0]/self.w[2,0]
        x2 = np.nan_to_num(x2)
        plt.plot(self.x1,x2,c=c,alpha=alpha,lw=lw)
        
    def activation_function(self, u):
        return 1 if u>=0 else -1
    
    def eqm(self):
        s = 0
        for k in range(self.N):
            x_k = self.X_train[:,k].reshape(self.p+1,1)
            u_k = (self.w.T@x_k)[0,0]
            d_k = self.d[k]
            s += (d_k - u_k)**2
        return s/(2*self.N)
    
    def fit(self):
        epochs = 0
        EQM1 = 0
        EQM2 = 1
        hist_eqm = []
        while epochs < self.max_epoch and abs(EQM1-EQM2)>self.tol:
            EQM1 = self.eqm()
            hist_eqm.append(EQM1)
            for k in range(self.N):
                x_k = self.X_train[:,k].reshape(self.p+1,1)
                u_k = (self.w.T@x_k)[0,0]
                d_k = self.d[k]
                e_k = d_k - u_k
                self.w = self.w + self.lr * e_k * x_k
            EQM2 = self.eqm()
            plt.pause(.1)
            self.draw_line(c='r',alpha=.5)
            epochs+=1
        plt.pause(.1)
        hist_eqm.append(EQM2)
        self.draw_line(c='g',alpha=1,lw=4)
        
        plt.figure(2)
        plt.plot(hist_eqm)
        plt.grid()
        plt.title("Curva de Aprendizado")
        plt.xlabel("Épocas")
        plt.ylabel("EQM")
        plt.show()

        


bp=1