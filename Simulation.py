from contextlib import nullcontext
import os

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'

# energy
def E(gamma, rho, p, v):
    return (p / (gamma - 1)) + (0.5 * rho * (v ** 2))

# pressure
def P(gamma, rho, v, E):
    return (gamma - 1) * (E - (0.5 * rho * (v ** 2)))

# speed of sound
def c_s(gamma, P, rho):
    return np.sqrt(gamma * P / rho)

class HLL_Solver():
    def __init__(self, gamma, dx, res):
        self.gamma = gamma
        self.dx = dx
        self.res = res

    # returns (lambda_plus, lambda_minus)
    def lambdas(self, U):
        v = U[1] / U[0]
        rho, E = U[0], U[2]
        cs = c_s(self.gamma, P(self.gamma, rho, v, E), rho)
        return (v + cs, v - cs)

    # returns (alpha_p, alpha_m)
    def alphas(self, U_L, U_R):
        lambda_L = self.lambdas(U_L)
        lambda_R = self.lambdas(U_R)
        alpha_p = max(0, lambda_L[0], lambda_R[0])
        alpha_m = max(0, -lambda_L[1], -lambda_R[1])

        return (alpha_p, alpha_m)

    # HLL flux
    def F_HLL(self, F_L, F_R, U_L, U_R):
        a_p, a_m = self.alphas(U_L, U_R)
        return (a_p * F_L + a_m * F_R - (a_p * a_m * (U_R - U_L))) / (a_p + a_m)

    def L(self, F_L, F_R):
        return - (F_R - F_L) / self.dx

    def solve(self, U, F):
        L_ = np.array([np.zeros(3) for _ in range(len(U))])
        
        # compute HLL flux at each interface
        for i in range(len(U)):
            F_L = self.F_HLL(F[i-1 if i > 0 else 0], F[i], 
                        U[i - 1 if i > 0 else 0], U[i])
            F_R = self.F_HLL(F[i], F[i + 1 if i < self.res - 1 else self.res - 1], 
                        U[i], U[i + 1 if i < self.res else self.res - 1] )
            
            # compute semi discrete L
            L_[i] = self.L(F_L, F_R)

        return L_

class PLM_Solver():
    def __init__(self, gamma, dx, res):
        self.gamma = gamma
        self.dx = dx
        self.res = res
        self.theta = 1.5

    def minmod(self, x, y, z):
        return 0.25 * abs(np.sign(x) + np.sign(y)) * (np.sign(x) + np.sign(z)) * min(abs(x), abs(y), abs(z))

    # calculates left-biased c value at cell interface to the left of i
    def c_L(self, C, i, j):
        # typo in hydro code guide? should be - not +
        return C[i if i >= 0 else 0][j] - (0.5 * self.minmod(self.theta * (C[i if i >= 0 else 0][j] - C[i-1 if i > 0 else 0][j]), 
                                    0.5 * (C[i+1 if i < len(C) - 1 else len(C) - 1][j] - C[i-1 if i > 0 else 0][j]), 
                                    self.theta * (C[i+1 if i < len(C) - 1 else len(C) - 1][j] - C[i if i >= 0 else 0][j])))

    # calculates right-biased c value at cell interface to the right of i
    def c_R(self, C, i, j):
        return C[i+1 if i < len(C) - 1 else len(C) - 1][j] - \
                (0.5 * self.minmod(self.theta * (C[i+1 if i < len(C) - 1 else len(C) - 1][j] - C[i if i >= 0 else 0][j]), 
                            0.5 * (C[i+2 if i < len(C) - 2 else len(C) - 1][j] - C[i if i >= 0 else 0][j]), 
                            self.theta * (C[i+2 if i < len(C) - 2 else len(C) - 1][j] - C[i+1 if i < len(C) - 1 else len(C) - 1][j])))

    # convert C vector to U vector
    def C_to_U(self, C):
        p, rho, v = C[0], C[1], C[2]
        return np.array([rho, rho * v, E(self.gamma, rho, p, v)])

    # convert U vector to C vector
    def U_to_C(self, U):
        rho, v, E = U[0], U[1] / U[0], U[2]
        return np.array([P(self.gamma, rho, v, E), rho, v])

    # piecewise linear method
    def solve(self, U, F):
        L_ = np.array([np.zeros(3) for _ in range(len(U))])

        # pressure, density, velocity
        C = np.array([np.zeros(3) for _ in range(len(U))])
        for i in range(len(U)):
            C[i] = self.U_to_C(U[i])
        
        # compute HLL flux at each interface
        for i in range(len(U)):
            # left interface
            U_L_L = self.C_to_U(np.array([self.c_L(C, i-1, j) for j in range(3)]))
            U_L_R = self.C_to_U(np.array([self.c_R(C, i-1, j) for j in range(3)]))

            # right interface
            U_R_L = self.C_to_U(np.array([self.c_L(C, i, j) for j in range(3)]))
            U_R_R = self.C_to_U(np.array([self.c_R(C, i, j) for j in range(3)]))
            
            F_L = self.F_HLL(F[i-1 if i > 0 else 0], F[i], 
                        U_L_L, U_L_R)
            F_R = self.F_HLL(F[i], F[i + 1 if i < len(U) - 1 else len(U) - 1], 
                        U_R_L, U_R_R)
            
            # compute semi discrete L
            L_[i] = self.L(F_L, F_R)

        return L_

class Simulation_1D():
    def __init__(self, gamma=1.4, resolution=100, dt=0.001, method="HLL"):
        self.gamma = gamma
        self.res = resolution
        self.x = np.linspace(0, 1, num=resolution, endpoint=False)
        self.dx = 1 / resolution
        self.dt = dt

        # conservative variable
        self.U = np.array(np.zeros(3) for _ in self.x)
        # flux
        self.F = np.array(np.zeros(3) for _ in self.x)
       
        if method == "HLL":
            self.solver = HLL_Solver(gamma, self.dx, self.res)
        elif method == "PLM":
            self.solver = PLM_Solver(gamma, self.dx, self.res) 
        else:    
            print("Invalid method provided: Must be HLL or PLM.")
            return

    # call in a loop to print dynamic progress bar
    def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        # print new line on complete
        if iteration == total: 
            print()

    # returns rho, v, p, E
    def get_vars(self):
        rho, v, E = self.U[:, 0], self.U[:, 1] / self.U[:, 0], self.U[:, 2]
        p = P(self.gamma, rho, v, E)	
        return rho, v, p, E	

    def compute_flux(self):
        rho, v, p, E = self.get_vars()
        self.F = np.array([rho * v, rho * (v ** 2) + p, (E + p) * v]).T

    def first_order_step(self):
        L = self.solver.solve(self.U, self.F)
        return np.add(self.U, L * self.dt)
   
    def high_order_step(self):
        """
        Third-order Runge-Kutta method
        """
        L_ = self.solver.solve(self.U, self.F)
        U_1 = np.add(self.U, L_ * self.dt)
        
        L_1 = self.solver.solve(U_1, self.F)
        U_2 = np.add(np.add((3/4) * self.U, (1/4) * U_1), (1/4) * self.dt * L_1)

        L_2 = self.solver.solve(U_2, self.F)
        return np.add(np.add((1/3) * self.U, (2/3) * U_2), (2/3) * self.dt * L_2) 