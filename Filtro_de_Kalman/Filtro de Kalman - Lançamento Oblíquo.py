import numpy as np

class Kalman:
    def __init__ (self, dt):
        self.F = np.matrix([[1, dt], [0, 1]])
        self.G = np.matrix([[0.5*dt**2], [dt]])
        self.u = -9.81
        self.H = np.matrix([1, 0])
        self.Q = np.matrix([[0.1, 0.1], [0.1, 0.1]])
        self.R = 25
        
    def state_prediction (self, x): # f[x(k|k)] = x(k+1|k)
        return self.F*x+self.G*self.u

    def state_measurement_prediction (self, x): # f[x(k+1|k)] = z(k+1|k)
        return self.H*self.state_prediction(x)

    def state_measurement_residual (self, x, z): #f[z(k+1), z(k+1|k)] = v(k+1)
        return z - self.state_measurement_prediction(x)

    def state_update (self, x, P, z): # f[x(k+1|k), K(k+1), v(k+1)] = x(k+1|k+1)
        K = self.filter_gain(P)
        return self.state_prediction(x)+K*self.state_measurement_residual(x, z)

    def covariance_prediction (self, P): #f[P(k|k)] = P(k+1|k)
        return self.F*P*self.F.T+self.Q

    def covariance_measurement_prediction (self, P): # f[P(k+1|k)] = S(k+1)
        return self.H*self.covariance_prediction(P)*self.H.T+self.R

    def filter_gain (self, P): # f[P(k+1|k), S(k+1)] = K(k+1)
        S = self.covariance_measurement_prediction(P)
        return self.covariance_prediction(P)*self.H.T*S.I

    def covariance_update (self, P): # f[K(k+1)] = P(k+1|k+1)
        K = self.filter_gain(P)
        S = self.covariance_measurement_prediction(P)
        return self.covariance_prediction(P)-K*S*K.T

    def filter (self, x, P, z):
        return (self.state_update(x, P, z), self.covariance_update(P))

# Gerando Medidas
#'''
y0 = 0
vy0 = 5
g = 9.81

t_0 = 0
t_f = 1
d_t = 1E-3
N = int((t_f-t_0)/d_t)

T = np.linspace(t_0, t_f, N)

Y, V_Y = [y0], [vy0]
for i in range(1, N):
    Y.append(max(Y[-1]+V_Y[-1]*d_t-(g*d_t**2)/2, 0))
    V_Y.append(V_Y[-1]-g*d_t)
Y_R, VY_R = list(Y), list(V_Y)
Y, V_Y = list(Y+0.1*np.random.normal(0, 1, N)), list(V_Y+1*np.random.normal(0, 1, N))
# Fim das Medidas
#'''

'''
with open("dados.txt") as f:
    T, Y, V_Y = list(), list(), list()
    for line in f.readlines():
        t, y, vy = line[:-1].split()
        T.append(float(t))
        Y.append(float(y))
        V_Y.append(float(vy))
        
d_t = T[1]
'''

Z = list(Y)
#Z = [np.matrix([[y], [vy]]) for y, vy in zip(Y, V_Y)]
                           

x = [np.matrix([[Y[0]], [V_Y[0]]])]
#x = [np.matrix([[1.5], [5]])]
P = [np.matrix([[10, 0], [10, 0]])]

kf = Kalman(d_t)

for z in Z[1:]:
    x_k, P_k = kf.filter(x[-1], P[-1], z)
    x.append(x_k)
    P.append(P_k)

import matplotlib.pyplot as plt

Y_U = [X.tolist()[0][0] for X in x]
VY_U = [X.tolist()[1][0] for X in x]

figure = plt.figure("Filtro de Kalman - Lançamento Vertical")
plot1 = figure.add_subplot(211)
plot1.plot(T, Y, label = 'Medidas')
plot1.plot(T, Y_U, label = 'Medidas Filtradas')
#plot1.plot(T, Y_R, label = 'Modelo Real')
plot1.legend()
plot2 = figure.add_subplot(212)
plot2.plot(T, V_Y, label = 'Medidas')
plot2.plot(T, VY_U, label = 'Medidas Filtradas')
#plot2.plot(T, VY_R, label = 'Modelo Real')
plot2.legend()
plt.show()
