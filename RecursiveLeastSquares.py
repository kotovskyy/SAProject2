import numpy as np

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
