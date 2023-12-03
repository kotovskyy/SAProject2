import numpy as np
import matplotlib.pyplot as plt

class RecursiveLeastSquares:
    def __init__(self, A0, P0, x0, x1):
        self.A = [A0,]
        self.P = [P0,]
        self.error = []
        self.n = 0
        self.Phi = np.array([[0, x1, x0]])

    def predict(self, x, y, lmbd=1):
        self.Phi[0][0] = x
        
        P_u = self.P[self.n] @ self.Phi.T @ self.Phi @ self.P[self.n]
        P_d = lmbd + self.Phi @ self.P[self.n] @ self.Phi.T
        P = (self.P[self.n] - P_u * (1/P_d))/lmbd
        
        
        err = y - self.Phi @ self.A[self.n]
        correction = P @ self.Phi.T @ err
    
        A = self.A[self.n] + correction

        self.A.append(A)
        self.P.append(P)
        self.error.append(err)
        
        self.Phi[0][2] = self.Phi[0][1]
        self.Phi[0][1] = self.Phi[0][0]
        
        self.n += 1
  
            
N = 1000
N_array = np.arange(N)

X = np.random.random(N)+1

a0 = 1
a1 = 10
a2 = -5
A = [a0, a1, a2]


V = np.zeros(N)
V[0] = X[0] * A[0]
V[1] = X[1] * A[0] + X[0] * A[1]
for i in range(2, N):
    V[i] = X[i] * A[0] + X[i-1] * A[1] + X[i-2] * A[2]

# odchylenie standardowe szumu
noise_stdev = 0.1
# szum z rozkładu normalnego
noise = noise_stdev * np.random.randn(N)

Y = V + noise

# plt.figure()
# plt.plot(N_array, V, label="v")
# plt.scatter(N_array, Y, color="r", marker=".", label="y")
# plt.legend()
# plt.grid(True)
# plt.show()

A0 = np.zeros(3)
P0 = 1000 * np.eye(3, 3)

RLS = RecursiveLeastSquares(A0, P0, X[0], X[1])
for i in range(2, N):
    x = X[i]
    y = Y[i]
    RLS.predict(x, y, lmbd=0.985)


estimate1 = []
estimate2 = []
estimate3 = []

for i in range(N-2):
    est = RLS.A[i]
    estimate1.append(est[0])
    estimate2.append(est[1])    
    estimate3.append(est[2])
    
    
est1true = A[0] * np.ones(N)
est2true = A[1] * np.ones(N)
est3true = A[2] * np.ones(N)

print(f"ESTIMATES : [{estimate1[-1]}, {estimate2[-1]}, {estimate3[-1]}]")

plt.figure()
plt.plot(N_array[:N-2], estimate1, label="a1_e")
plt.plot(N_array[:N-2], estimate2, label="a2_e")
plt.plot(N_array[:N-2], estimate3, label="a3_e")
plt.axhline(y=A[0], color='r', linestyle='--', label="a1*")
plt.axhline(y=A[1], color='g', linestyle='--', label="a2*")
plt.axhline(y=A[2], color='b', linestyle='--', label="a3*")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Value")

plt.figure()
plt.plot(N_array, est1true, label="a1*", linewidth=2)
plt.plot(N_array[:N-2], estimate1, label="a1_e", color="r", linewidth=0.9)

plt.show()
