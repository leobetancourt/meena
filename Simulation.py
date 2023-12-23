from contextlib import nullcontext
import os

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'

class Simulation_1D():
	def __init__(self, gamma=1.4, num_cells=100, dt=0.001):
		self.gamma = gamma
		self.num_cells = num_cells
		self.dt = dt
		self.dx = 1 / num_cells

	# variable conversion methods
	# energy
	def E(self, rho, p, v):
		return (p / (self.gamma - 1)) + (0.5 * rho * (v ** 2))

	# pressure
	def P(self, rho, v, E):
		return (self.gamma - 1) * (E - (0.5 * rho * (v ** 2)))
	
	# speed of sound
	def c_s(self, P, rho):
		return np.sqrt(self.gamma * P / rho)

	# returns (lambda_plus, lambda_minus)
	def lambdas(self, U):
		v = U[1] / U[0]
		rho, E = U[0], U[2]
		cs = self.c_s(self.gamma, self.P(rho, v, E), rho)
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

	def HLL(self, U, F):
		L_ = np.array([np.zeros(3) for _ in range(len(U))])
		
		# compute HLL flux at each interface
		for i in range(len(U)):
			F_L = self.F_HLL(F[i-1 if i > 0 else 0], F[i], 
						U[i - 1 if i > 0 else 0], U[i])
			F_R = self.F_HLL(F[i], F[i + 1 if i < len(U) - 1 else len(U) - 1], 
						U[i], U[i + 1 if i < len(U) - 1 else len(U) - 1] )
			
			# compute semi discrete L
			L_[i] = self.L(F_L, F_R)

		return L_
		
