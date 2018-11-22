#Allow import of A_FMM from repository folder, if A_FMM is already in path can be removed
import sys
sys.path.append('../.')
#Other imports
import numpy as np
import A_FMM
from matplotlib.backends.backend_pdf import PdfPages


#Defining data
Nx=30                             #Set truncation order in X direction
k0_l=np.linspace(0.01,2.0,500)    #Define vector for sweep over energies (in unit of a/lambda)


#Defining creator
cr=A_FMM.creator()                #Define the creator
cr.slab(12.0,2.0,2.0,0.3)         #define the structure: inputs of slab are eps core, eps lower cladding, eps upper cladding, width
wave=A_FMM.layer(Nx,0,cr)         #Define layer istance: inputs are the truncation orders in x and y, and a creator istance
wave.trasform(ex=0.8)             #Add coordinate transform for supercell (ex is the width of the unmpped region --- unit of cell width)
wave.eps_plot('slab')             #Plot reconstructed eps in slab.pdf file

for k0 in k0_l:                                       #Loop over energies
    wave.mode(k0)                                     #call mode method of layer class to solve the mode
    neff=np.flip(np.sort(wave.gamma)[-10:].real,0)    #effective indexes are stored in the vector gamma of the layer class. The biggest ten are retained 
    print 11*'%15.8f' % ((k0,)+tuple(neff))           #Print results to screen




