import matplotlib.pyplot as plt
from ergodic_coveragetwo import ErgCover
import jax.numpy as op
import pickle
import numpy as np

pickle_file_path = 'build_prob/random_maps/random_map_0.pickle'
data=pickle.load(open(pickle_file_path,'rb'))
map=data['pdfs'][1]
npix=map.shape[0]

n_scalar = 10 # number of terms in fourrier series
nS=3  # number of steps to be taken
nA= 50
step_size=20

pbm_file = "build_prob/random_maps/random_map_0.pickle"

g=np.zeros(2) # initial state

pdf = np.zeros((100,100))
pdf=np.copy(map)
pdf = np.asarray(pdf.flatten())


erg,tr = ErgCover(pdf,1,nS,100, g, n_scalar, npix, 2000, None, grad_criterion=True,direct_FC=None,stepize=step_size)



x=[]
y=[]
for i in range(len(tr)):
    x.append(tr[i][0]*100)
    y.append(tr[i][1]*100)
plt.scatter(x,y)
plt.imshow(pdf.reshape(100,100))
plt.show()

print("ergodicity",erg)