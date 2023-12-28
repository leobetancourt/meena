from contextlib import nullcontext
import os

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'

class HLL_solver():
	def __init__(self, gamma):
		pass

class Simulation_1D():
	def __init__(self, gamma=1.4, num_cells=100, dt=0.001, method="HLL"):
		self.gamma = gamma
		if method == "HLL" or method == "PLM":
			self.method = method
		else:
			print("Invalid method provided: must be HLL or PLM.")
			return
		
		self.num_cells = num_cells
		self.x = np.linspace(0, 1, num=num_cells, endpoint=False)
		self.dx = 1 / num_cells
		self.dt = dt

		# conservative variable
		self.U = np.array(np.zeros(3) for _ in self.x)
		# flux
		self.F = np.array(np.zeros(3) for _ in self.x)

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
		p = self.P(rho, v, E)	
		return rho, v, p, E	

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

	def HLL(self):
		L_ = np.array([np.zeros(3) for _ in range(len(self.U))])
		
		# compute HLL flux at each interface
		for i in range(len(self.U)):
			F_L = self.F_HLL(self.F[i-1 if i > 0 else 0], self.F[i], 
						self.U[i - 1 if i > 0 else 0], self.U[i])
			F_R = self.F_HLL(self.F[i], self.F[i + 1 if i < self.res - 1 else self.res - 1], 
						self.U[i], self.U[i + 1 if i < self.res else self.res - 1] )
			
			# compute semi discrete L
			L_[i] = self.L(F_L, F_R)

		return L_

	def compute_flux(self):
		rho, v, p, E = self.get_vars()
		self.F = np.array([rho * v, rho * (v ** 2) + p, (E + p) * v]).T

	def first_order_step(self):
		L = self.HLL() if self.method == "HLL" else self.PLM()
		return np.add(self.U, L * self.dt)
