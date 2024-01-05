import numpy as np
from jax import vmap, jit, grad
import jax.numpy as jnp
from jax.lax import scan
import copy
# from jax.config import config; config.update("jax_enable_x64", True)
from functools import partial
step_size=50
tempvar=1
temptheta=0
rob_vel = 0.8

def wrapToPi(x):
	if x > 3.14:
		x = x - 2*3.14
	elif x < -3.14:
		x = x + 2*3.14
	return x

def fDyn(x, u): # dynamics of the robot - point mass
	xnew = x + jnp.array([jnp.tanh(u[0]),jnp.tanh(u[0]),10*u[1]])
	# xnew = x + jnp.array([0.8,0.8,10*u[1]])
	return xnew, x

def fDiffDrive(x0, u):
	"""
	x0 = (x,y,theta)
	u = (v,w)

	x0 = (x,y,theta,v)
	u = (w,a)
	"""
	u = jnp.tanh(u) #Limit the maximum velocity to 1
	x = x0 + jnp.array([jnp.cos(x0[2])*0.02, jnp.sin(x0[2])*0.02, 0.8*u[0]])

	return x, x0

def get_hk(k): # normalizing factor for basis function
	_hk = jnp.array((2. * k + np.sin(2 * k))/(4. * k))
	_hk = _hk.at[np.isnan(_hk)].set(1.)	
	return np.sqrt(np.prod(_hk))

def fk(x, k): # basis function
    return jnp.prod(jnp.cos(x*k))

def GetTrajXY(u, x0):
	"""
	"""

	xf, tr0 = scan(fDiffDrive, x0, u)
	some=tr0[step_size-1][2]

	tr = tr0[:,0:2] # take the (x,y) part of all points
	return xf, tr,some

def GetTrajXYTheta(u,x0):
	xf, tr = scan(fDiffDrive, x0, u)
	return xf, tr


class ErgCalc(object):
	"""
	modified from Ian's Ergodic Coverage code base.
	"""
	def __init__(self, pdf, n_agents, nA, n_fourier, nPix):
		# print("Number of agents: ", n_agents)
		self.n_agents = n_agents
		self.nPix = nPix
		self.nA = nA
		# aux func
		self.fk_vmap = lambda _x, _k: vmap(fk, in_axes=(0,None))(_x, _k)
		self.x0=jnp.array(())

		# fourier indices
		k1, k2 = jnp.meshgrid(*[jnp.arange(0, n_fourier, step=1)]*2)
		k = jnp.stack([k1.ravel(), k2.ravel()]).T
		self.k = jnp.pi*k

		# lambda, the weights of different bands.
		self.lamk = (1.+jnp.linalg.norm(self.k/jnp.pi,axis=1)**2)**(-4./2.)

		# the normalization factor
		hk = []
		for ki in k:
			hk.append(get_hk(ki))
		self.hk = jnp.array(hk)

		# compute phik
		if isinstance(nPix,int) == True:
			X,Y = jnp.meshgrid(*[jnp.linspace(0,1,num=self.nPix)]*2)
		else: #Using this when using a window around the agent and the window is not a square
			X,Y = jnp.meshgrid(jnp.linspace(0,1,num=self.nPix[0]),jnp.linspace(0,1,num=self.nPix[1]))
		_s = jnp.stack([X.ravel(), Y.ravel()]).T
		# print("nPix: ", self.nPix)
		# print("Shape of vmap: ",vmap(self.fk_vmap, in_axes=(None, 0))(_s, self.k).shape)
		# print("Shape of pdf: ", pdf.shape)
		
		phik = jnp.dot(vmap(self.fk_vmap, in_axes=(None, 0))(_s, self.k), pdf) #vmap(p)(_s)
		phik = phik/phik[0]
		self.phik = phik/self.hk		  

		# for reconstruction
		self.phik_recon = jnp.dot(self.phik, vmap(self.fk_vmap, in_axes=(None, 0))(_s, self.k)).reshape(X.shape)
		
		# to compute gradient func
		self.gradient = jit(grad(self.fourier_ergodic_loss))
		self.temptraj=[]
		self.full_u=[]
		self.fulltraj=[]
		self.step=0
		self.step_size=50
		self.precau=[]

		return
	
	def get_recon(self, FC):
		X,Y = jnp.meshgrid(*[jnp.linspace(0,1,num=self.nPix)]*2)
		_s = jnp.stack([X.ravel(), Y.ravel()]).T
		return jnp.dot(FC, vmap(self.fk_vmap, in_axes=(None, 0))(_s, self.k)).reshape(X.shape)

	def get_ck(self, tr):
		"""
		given a trajectory tr, compute fourier coeffient of its spatial statistics.
		k is the number of fourier coeffs.
		"""
		ck = jnp.mean(vmap(partial(self.fk_vmap, tr))(self.k), axis=1)
		ck = ck / self.hk
		return ck

	def fourier_ergodic_loss(self, u): 
		ck = 0
		trajectories=copy.copy(self.fulltraj)
		x0_i=self.x0
		xf, tr,thet = GetTrajXY(u, x0_i)
		self.temptehta=thet
		for k in range(len(tr)):
			trajectories.append(tr[k])
		ck = self.get_ck(jnp.array(trajectories))
		self.temptraj=trajectories
		self.u=u

		
		traj_cost = 0 
		# lambda, the weights of different bands.
		traj_cost += jnp.mean((jnp.array(trajectories) - jnp.array([0.5,0.5]))**8)
		ergodicity = jnp.sum(self.lamk*jnp.square(self.phik - ck)) + 3e-2 * jnp.mean(u**2) + traj_cost
		return ergodicity



	def trajstep(self,flag):
		self.step+=1
		# tempvar=tempvar+1

		# for i in range(len(self.temptraj)):
		self.fulltraj=[]
		# print(len(self.temptraj)," fcc")
		if (flag):
			for i in range(len(self.u)):
				self.full_u.append(self.u[i])
			for i in range(len(self.temptraj)):
				self.fulltraj.append(self.temptraj[i])
			return
		for i in range(self.step_size):
			self.full_u.append(self.u[i])
		for i in range(self.step_size*self.step):
			a=max(self.temptraj[i][0],0)
			a=min(self.temptraj[i][0],1)
			b=max(self.temptraj[i][1],0)
			b=min(self.temptraj[i][1],1)

			if a<0:
				a=0
			if a>1:
				a=1

			if b<0:
				b=0
			if b>1:
				b=1
				

			self.fulltraj.append([a,b])
			# print(a,b)
		self.theta=wrapToPi(self.temptehta)
		# print(self.precau[self.step_size-1], "   theta")
		# self.theta=self.precau[self.step_size-1][2]
		return





	def fourier_ergodic_loss_traj(self,traj):
		ck = self.get_ck(traj)
		traj_cost = jnp.mean((traj - jnp.array([0.5,0.5]))**8)
		ergodicity = jnp.sum(self.lamk*jnp.square(self.phik - ck)) + traj_cost
		return ergodicity

	def traj_stat(self, u, x0):
		"""
		"""
		xf, tr = GetTrajXY(u, x0)
		ck = self.get_ck(tr)
		X,Y = jnp.meshgrid(*[jnp.linspace(0,1,num=self.nPix)]*2)
		_s = jnp.stack([X.ravel(), Y.ravel()]).T
		pdf = jnp.dot(ck, vmap(self.fk_vmap, in_axes=(None, 0))(_s, self.k)).reshape(X.shape)
		return pdf
