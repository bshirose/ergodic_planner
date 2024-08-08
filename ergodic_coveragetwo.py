import jax.numpy as np
from jax.experimental import optimizers
import random
from jax import vmap, jit, grad
# from jax.config import config; config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import ergodic_metrictwo

GLOBAL_NUM_K = 0

def ErgCover(pdf, n_agents,nS, nA, s0, n_fourier, nPix, nIter, u_init=None, grad_criterion=False,direct_FC=None, stepize=50):
	"""
	run ergodic coverage over a info map. Modified from Ian's code.
	return a list of control inputs.
	"""
	step_size=stepize
	# print("****************************************************************")
	# print("[INFO] ErgCover, nA =", nA, " s0 =", s0, " n_fourier =", n_fourier, " stop_eps =", stop_eps)
	
	
	if direct_FC is not None:
		erg_calc = ergodic_metrictwo.ErgCalc(pdf, n_agents, 1000, n_fourier, nPix,step_size)
		erg_calc.phik = direct_FC
		erg_calc.phik = erg_calc.phik/erg_calc.phik[0]
	else:
		erg_calc = ergodic_metrictwo.ErgCalc(pdf, n_agents, 1000, n_fourier, nPix,step_size)

	opt_init, opt_update, get_params = optimizers.adam(1e-3) #Declaring Adam's optimizer

	# initial conditions
	
	s0 = np.array(s0)
	erg_calc.x0=s0
	u = np.zeros((nA,n_agents*2))
	u=np.array(u)




	if u_init is not None:
		u = np.array(u_init)
	opt_state = opt_init(u)
	log = []

	i = 0
	steps=nS
	
	for s in range (steps):
		for i in range(nIter):
			g = erg_calc.gradient(u)
			opt_state = opt_update(i, g, opt_state)
			u = get_params(opt_state)
			# log.append(erg_calc.fourier_ergodic_loss(u).copy())
			# print( np.linalg.norm(g), " norm of gradient")
			# print(i," iteration",erg_calc.fourier_ergodic_loss(u))
			# if grad_criterion:
			# 	if -0.000001 < np.linalg.norm(g) < 0.000001:
			# 		print("Reached grad criterion at iteration: ", i)
			# 		break
		
		t=False

		e=erg_calc.fourier_ergodic_loss(u)


		if s==steps-1:
			t=True
		erg_calc.trajstep(t)
		erg_calc.gradient = jit(grad(erg_calc.fourier_ergodic_loss))
		# print("****step: ",s)
		for som in range(nA):
			u=u.at[som,0].set(0)
		
		for i in range(n_agents):
			erg_calc.x0 = erg_calc.x0.at[2*i].set(erg_calc.fulltraj[-1][2*i])
			erg_calc.x0 = erg_calc.x0.at[2*i+1].set(erg_calc.fulltraj[-1][2*i+1])

		# print(erg_calc.x0," x0")


	tr=erg_calc.fulltraj

	# if ifDisplay : # final traj
	# 	plt.figure(figsize=(5,5))
		
	# 	X,Y = np.meshgrid(*[np.linspace(0,1,num=nPix)]*2)
	# 	plt.contourf(X, Y, erg_calc.phik_recon, levels=np.linspace(np.min(erg_calc.phik_recon), np.max(erg_calc.phik_recon),100), cmap='gray')
	# 	for o in range(len(tr)):
	# 		plt.plot(tr[o][0],tr[o][1], "r.:")
	# 	for o in range(len(tr)):
	# 		plt.plot(tr[o][2],tr[o][3], "b.:")
	# 	plt.axis("off")
	# 	plt.pause(1)
	return e,tr