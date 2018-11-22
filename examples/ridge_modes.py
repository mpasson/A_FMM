#Allow import of A_FMM from repository folder, if A_FMM is already in path can be removed
import sys
sys.path.append('../.')
#Other imports
import numpy as np
import A_FMM
from matplotlib.backends.backend_pdf import PdfPages


#Defining data
Nx=10                                #Set truncation order in X direction
Ny=10                                #Set truncation order in X direction
k0_l=np.linspace(0.01,2.0,500)       #Define vector for sweep over energies (in unit of a/lambda)


#Defining creator
cr=A_FMM.creator()                      #Define the creator
cr.ridge(12.0,2.0,2.0,0.4,0.31,0.15)    #define the structure: inputs of slab are eps core, eps lower cladding, eps upper cladding, width, heigth, heigth cladding
ridge=A_FMM.layer(Nx,Ny,cr)             #Define layer istance: inputs are the truncation orders in x and y, and a creator istance
ridge.trasform(ex=0.8,ey=0.8)           #Add coordinate transform for supercell --- both ex and ey (width of unmapped region) has to be specified
ridge.eps_plot('ridge')                 #Plot reconstructed eps in ridge.pdf file

for k0 in k0_l:                                         #Loop over energies
    ridge.mode(k0)                                      #call mode method of layer class to solve the mode
    print 11*'%15.8f' % ((k0,)+tuple(neff))             #effective indexes are stored in the vector gamma of the layer class. The biggest ten are retained 
    neff=np.flip(np.sort(ridge.gamma)[-10:].real,0)     #Print results to screen 


