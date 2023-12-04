import numpy as onp
import jax.numpy as np
from jax.experimental import optimizers
import random
from jax import vmap, jit, grad
# from jax.config import config; config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import ergodic_metric
from scipy.optimize import minimize

GLOBAL_NUM_K = 0

def ErgCover(pdf, n_agents, nA, s0, n_fourier, nPix, nIter, ifDisplay, u_init=None, stop_eps=-1, kkk=0,grad_criterion=False,direct_FC=None):
	"""
	run ergodic coverage over a info map. Modified from Ian's code.
	return a list of control inputs.
	"""
	# print("****************************************************************")
	# print("[INFO] ErgCover, nA =", nA, " s0 =", s0, " n_fourier =", n_fourier, " stop_eps =", stop_eps)
	
	
	if direct_FC is not None:
		erg_calc = ergodic_metric.ErgCalc(pdf, n_agents, 1000, n_fourier, nPix)
		erg_calc.phik = direct_FC
		erg_calc.phik = erg_calc.phik/erg_calc.phik[0]
	else:
		erg_calc = ergodic_metric.ErgCalc(pdf, n_agents, 1000, n_fourier, nPix)

	opt_init, opt_update, get_params = optimizers.adam(1e-3) #Declaring Adam's optimizer

	# initial conditions
	
	s0 = np.array(s0)
	erg_calc.x0=s0
	# print(x0.shape," shape")
	# print("Initial state of the agents: ", x0)
	
	u = np.zeros((nA*n_agents,1))
	# for i in range(nA):
	# 	# u[i][0]=random.random()/100
	# 	u = u.at[i,0].set(random.random()/100)

	# u = onp.random.rand(nA * n_agents, 2)
	# print(u.max()," max")
	# u=u/100
	
	
	u=np.array(u)




	if u_init is not None:
		u = np.array(u_init)
	opt_state = opt_init(u)
	log = []

	# if stop_eps > 0:
	# 	nIter = int(1e5) # set a large number, stop until converge.

	# if grad_criterion == True: # We want to iterate till we find a minima 
	# 	nIter = int(10) # Again set a very large number of iterations
		# print("**Grad criterion activated!**")

# 	bounds = [(-1, 1) for _ in u]
# 	result = minimize(erg_calc.fourier_ergodic_lossbb, u)

# # Get the optimal u
# 	optimal_u = result.x
# 	log.append(erg_calc.fourier_ergodic_loss(optimal_u, x0).copy())




	i = 0
	steps=10
	for s in range (steps):
		for i in range(nIter):
			g = erg_calc.gradient(u)
			opt_state = opt_update(i, g, opt_state)
			u = get_params(opt_state)
			# log.append(erg_calc.fourier_ergodic_loss(u).copy())
			# print( np.linalg.norm(g), " norm of gradient")
			print(i," iteration")
			if grad_criterion:
				if -0.000001 < np.linalg.norm(g) < 0.000001:
					print("Reached grad criterion at iteration: ", i)
					break
		
		t=False
		# if abs(e-erg_calc.fourier_ergodic_loss(u))<0.0002:

		e=erg_calc.fourier_ergodic_loss(u)


		if s==steps-1:
			t=True
		erg_calc.trajstep(t)
		erg_calc.gradient = jit(grad(erg_calc.fourier_ergodic_loss))
		print("****step: ",s)
		for som in range(nA):
			if som>45:
				u=u.at[som,0].set(0)
				# u=u.at[som,1].set(0)
			else:
				u = u.at[som,0].set(u[som+45][0])
				# u = u.at[som,1].set(u[som+45][1])


		erg_calc.x0 = erg_calc.x0.at[0].set(erg_calc.fulltraj[-1][0])
		erg_calc.x0 = erg_calc.x0.at[1].set(erg_calc.fulltraj[-1][1])
		erg_calc.x0 = erg_calc.x0.at[2].set(erg_calc.theta)

		print(erg_calc.x0," x0")



















	# i = 0
	# steps=10
	# for s in range (steps):
	# 	for i in range(nIter):
	# 		# print("****Iter: ",i)
	# 		g = erg_calc.gradient(u, x0)
	# 		# print(g)
	# 		opt_state = opt_update(i, g, opt_state)
	# 		u = get_params(opt_state)
	# 		log.append(erg_calc.fourier_ergodic_loss(u, x0).copy())
	# 		print( np.linalg.norm(g), " norm of gradient")
	# 		# print( log[-1], " loss")
	# 		print(i," iteration")
	# 		## check for convergence
	# 		if grad_criterion: # at least 10 iterationss
	# 			if -0.000001 < np.linalg.norm(g) < 0.000001:
	# 				print("Reached grad criterion at iteration: ", i)
	# 				break
	# 	q=u[-1][0]
	# 	sign=0
	# 	if q>0:
	# 		sign=1
	# 	else:
	# 		sign=-1

	# 	t=False
	# 	if s==steps-1:
	# 		t=True
	# 	erg_calc.trajstep(t)
	# 	print("****step: ",s)
	# 	# for jok in range(100):
	# 	# print(u[0][0]," u[v][0] ",0)
	# 	# print(u[0][1]," u[s][1] ",0)
	# 	# print(u[49][0]," u[v][0] ",1)
	# 	# print(u[49][1]," u[s][1] ",1)

	# 	# u = np.zeros((nA*n_agents,2))
	# 	# u=np.array(u)
	# 	# for som in range(nA):
	# 	# 	u = u.at[som,0].set(sign*random.random()/500)
	# 	# opt_state = opt_init(u)
	# 	for som in range(nA):
	# 		if som>49:
	# 			u=u.at[som,0].set(0)
	# 			u=u.at[som,1].set(0)
	# 		else:
	# 			# print(som, " som")
	# 			# print(u[som][0]," u[som][0]")
	# 			u = u.at[som,0].set(u[som+49][0])
	# 			u = u.at[som,1].set(u[som+49][1])


	# 	x0 = x0.at[0].set(erg_calc.fulltraj[-1][0])
	# 	x0 = x0.at[1].set(erg_calc.fulltraj[-1][1])
	# 	x0 = x0.at[2].set(erg_calc.theta)
	# 	print(x0," x0")
	# 	# print(x0[2]," theta")
	# 	# x0[0]=erg_calc.fulltraj[-1][0]
	# 	# x0[1]=erg_calc.fulltraj[-1][1]
	# 	# x0[2]=erg_calc.theta
	# 		# elif i > 10 and stop_eps > 0: # at least 10 iterationss
	# 		# 	if onp.abs(log[-1]) < stop_eps:
	# 		# 		# print("Reached final threshold before number of iterations!")
	# 		# 		break
		
	# i = 0
	# steps=4
	# for s in range (steps):
	# 	for som in range(nA):
	# 		if som>0:
	# 			u=u.at[som,0].set(0.001)
	# 			u=u.at[som,1].set(0)
	# 	for i in range(nIter):
	# 		# print("****Iter: ",i)
	# 		# g = erg_calc.gradient(u, x0)
	# 		# # print(g)
	# 		# opt_state = opt_update(i, g, opt_state)
	# 		# u = get_params(opt_state)
	# 		log.append(erg_calc.fourier_ergodic_loss(u, x0).copy())
	# 		# print( np.linalg.norm(g), " norm of gradient")
	# 		# print( log[-1], " loss")
	# 		print(i," iteration")
	# 		## check for convergence
	# 		# if grad_criterion: # at least 10 iterationss
	# 		# 	if -0.000001 < np.linalg.norm(g) < 0.000001:
	# 		# 		print("Reached grad criterion at iteration: ", i)
	# 		# 		break
	# 	q=u[-1][0]
	# 	sign=0
	# 	if q>0:
	# 		sign=1
	# 	else:
	# 		sign=-1

	# 	t=False
	# 	if s==steps-1:
	# 		t=True
	# 	erg_calc.trajstep(t)
	# 	print("****step: ",s)
	# 	# for jok in range(100):
	# 	print(u[0][0]," u[v][0] ",0)
	# 	print(u[0][1]," u[s][1] ",0)
	# 	print(u[1][0]," u[v][0] ",1)
	# 	print(u[1][1]," u[s][1] ",1)

	# 	# u = np.zeros((nA*n_agents,2))
	# 	# u=np.array(u)
	# 	# for som in range(nA):
	# 	# 	u = u.at[som,0].set(sign*random.random()/500)
	# 	# opt_state = opt_init(u)
	# 	for som in range(nA):
	# 		if som>49:
	# 			u=u.at[som,0].set(0)
	# 			u=u.at[som,1].set(0)
	# 		else:
	# 			# print(som, " som")
	# 			# print(u[som][0]," u[som][0]")
	# 			u = u.at[som,0].set(u[som+49][0])
	# 			u = u.at[som,1].set(u[som+49][1])


	# 	x0 = x0.at[0].set(erg_calc.fulltraj[-1][0])
	# 	x0 = x0.at[1].set(erg_calc.fulltraj[-1][1])
	# 	x0 = x0.at[2].set(erg_calc.theta)
	# 	print(x0," x0")
	# 	# print(x0[2]," theta")
	# 	# x0[0]=erg_calc.fulltraj[-1][0]
	# 	# x0[1]=erg_calc.fulltraj[-1][1]
	# 	# x0[2]=erg_calc.theta
	# 		# elif i > 10 and stop_eps > 0: # at least 10 iterationss
	# 		# 	if onp.abs(log[-1]) < stop_eps:
	# 		# 		# print("Reached final threshold before number of iterations!")
	# 		# 		break
		 




	if ifDisplay : # final traj
		plt.figure(figsize=(5,5))
		tr=erg_calc.fulltraj
		X,Y = np.meshgrid(*[np.linspace(0,1,num=nPix)]*2)
		plt.contourf(X, Y, erg_calc.phik_recon, levels=np.linspace(np.min(erg_calc.phik_recon), np.max(erg_calc.phik_recon),100), cmap='gray')
		# plt.scatter(tr[:,0],tr[:,1], c='r', marker="*:")
		# plt.plot(tr[0,0],tr[0,1], "ro:")
		for o in range(len(tr)):
			plt.plot(tr[o][0],tr[o][1], "r.:")
		# plt.plot(tr[:,0],tr[:,1], "r.:")

			# plt.plot(tr1[0,0],tr1[0,1], "bo:")
			# plt.plot(tr1[:,0],tr1[:,1], "b.:")
		plt.axis("off")
		plt.pause(1)
	return get_params(opt_state), log, i,tr

