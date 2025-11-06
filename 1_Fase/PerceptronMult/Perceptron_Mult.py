import numpy as np
import matplotlib.pyplot as plt

class Perceptron_Mult:
    def __init__(self, layer_dims, X_train, y_train, learning_rate=1e-3, max_epoch=10000, criterio_parada=0.01, plot=True):
        self.p, self.N = X_train.shape
        self.X_train = np.vstack((-np.ones((1, self.N)), X_train))
        self.max_epoch = max_epoch
        self.lr = learning_rate
        self.criterio_parada = criterio_parada
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims)
        self.d = np.array(y_train).reshape(-1, 1)
        self.plot = plot
        self.errors_per_epoch = []
        self.W = []

        for j in range(1, self.num_layers):
            num_inputs_com_bias = layer_dims[j - 1] + 1
            num_outputs = layer_dims[j]
            w = np.random.rand(num_outputs, num_inputs_com_bias) - 0.5
            self.W.append(w)

        self.i = [None] * len(self.W)
        self.y = [None] * len(self.W)

        if plot:
            self.fig = plt.figure(2)
            self.ax = self.fig.add_subplot()
            self.ax.scatter(self.X_train[1, self.d[:, 0] == 1],
                            self.X_train[2, self.d[:, 0] == 1],c='r', marker='s', s=120, edgecolor='k')
            self.ax.scatter(self.X_train[1, self.d[:, 0] == -1],
                            self.X_train[2, self.d[:, 0] == -1],c='b', marker='o', s=120, edgecolor='k')
            margin = 0
            x_min, x_max = self.X_train[1].min() - margin, self.X_train[1].max() + margin
            y_min, y_max = self.X_train[2].min() - margin, self.X_train[2].max() + margin

            self.ax.set_xlim(x_min, x_max)
            self.ax.set_ylim(y_min, y_max)
            self.x1 = np.linspace(x_min, x_max)
            self.ax.set_xlabel("Variável 1")
            self.ax.set_ylabel("Variável 2")
            self.ax.set_title("Plano de Regressão do Perceptron")

    def activation_function(self, u):
        return np.where(u >= 0, 1, -1)

    def g_derivada(self, u):
        return np.ones_like(u) * 0.5

    def EQM(self):
        EQM_total = 0
        for k in range(self.N):
            x_k = self.X_train[:, k].reshape(self.p + 1, 1)
            self.forward(x_k)
            d_k = self.d[k].reshape(-1, 1)
            y_saida = self.y[-1]
            EQI = np.sum((d_k - y_saida) ** 2)
            EQM_total += EQI
        return EQM_total / (2 * self.N)

    def fit(self):
        EQM = 1
        Epoch = 0
        while EQM > self.criterio_parada and Epoch < self.max_epoch:
            for n in range(self.N):
                x_amostra = self.X_train[:, n].reshape(self.p + 1, 1)
                d = np.array(self.d[n], ndmin=2).reshape(-1, 1)
                self.forward(x_amostra)
                self.backward(x_amostra, d)
            EQM = self.EQM()
            self.errors_per_epoch.append(EQM)
            Epoch += 1
        

    def backward(self, x_amostra, d):
        delta = [None] * len(self.W)
        j = len(self.W) - 1
        while j >= 0:
            if j + 1 == len(self.W):
                delta[j] = self.g_derivada(self.i[j]) * (d - self.y[j])
                y_bias = np.vstack(([-1], self.y[j - 1]))
                self.W[j] += self.lr * (delta[j] @ y_bias.T)
            elif j == 0:
                Wb = self.W[j + 1].T[1:, :]
                delta[j] = self.g_derivada(self.i[j]) * (Wb @ delta[j + 1])
                self.W[j] += self.lr * (delta[j] @ x_amostra.T)
            else:
                Wb = self.W[j + 1].T[1:, :]
                delta[j] = self.g_derivada(self.i[j]) * (Wb @ delta[j + 1])
                y_bias = np.vstack(([-1], self.y[j - 1]))
                self.W[j] += self.lr * (delta[j] @ y_bias.T)
            j -= 1

    def forward(self, x_amostra):
        self.i = []
        self.y = []
        for j in range(len(self.W)):
            if j == 0:
                i_j = self.W[j] @ x_amostra
                y_j = self.activation_function(i_j)
            else:
                y_bias = np.vstack(([-1], self.y[j - 1]))
                i_j = self.W[j] @ y_bias
                y_j = self.activation_function(i_j)
            self.i.append(i_j)
            self.y.append(y_j)
        return self.y

    def predict(self, X_test):
        p_test, N_test = X_test.shape
        X_test_bias = np.vstack((-np.ones((1, N_test)), X_test))
        outputs = []
        for n in range(N_test):
            x = X_test_bias[:, n].reshape(p_test + 1, 1)
            y = self.forward(x)
            outputs.append(y[-1])
        return np.array(outputs)
