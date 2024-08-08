import matplotlib.pyplot as plt
from ergodic_coveragetwo import ErgCover
import jax.numpy as op
import pickle
import numpy as np

pickle_file_path = 'build_prob/random_maps/random_map_0.pickle' # path to the pickle file containing the pdf
data=pickle.load(open(pickle_file_path,'rb')) # load the pickle file
map=data['pdfs'][1] # get the pdf from the pickle file
npix=map.shape[0] # get the size of the map

n_scalar = 10 # number of terms in fourrier series
nS=3  # number of steps to be taken
nA= 100 # number of actions each agent will plan for at each time instance
step_size=20 # step size agent will travel in each time instance
n_agents=1 # number of agents
g=np.zeros(n_agents*2) # initial state of the agent (x,y)

pdf = np.zeros((100,100))
pdf=np.copy(map)
pdf = np.asarray(pdf.flatten())


erg,tr = ErgCover(pdf,n_agents,nS,nA, g, n_scalar, npix, 2000, None, grad_criterion=True,direct_FC=None,stepize=step_size)

#erg is the ergodicity value and tr is the trajectory of all the agents

x=[]
y=[]
for i in range(len(tr)):
    x.append(tr[i][0]*100)
    y.append(tr[i][1]*100)
plt.scatter(x,y)
plt.imshow(pdf.reshape(100,100))
plt.show()

print("ergodicity",erg)