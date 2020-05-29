#This example illustrates the use of the complex coordinate transform as PML boundary condition
#Follow the benchmark proposed in 'Jean Paul Hugonin and Philippe Lalanne, "Perfectly matched layers as nonlinear coordinate transforms: a generalized formalization," J. Opt. Soc. Am. A 22, 1844-1849 (2005)'

#Allow import of A_FMM from repository folder, if A_FMM is already in path can be removed
import sys
sys.path.append('../.')
#Other imports
import numpy as np
import A_FMM

#Initializing creator structure for layer creation
cr=A_FMM.creator()

#Computational cell for the layer is assumed 1D with period of 1 micron, only NX is set
NX=20
k0=1.0/0.975 #Setting energy (in unit of period/labda) (lambda_0=975 nm)

#Creting layers involved in structure
cr.slab(3.5**2,2.9**2,1.0,0.3) #Setting creator for waveguide
waveguide=A_FMM.layer(NX,0,cr) #Creating waveguide layer
cr.slab(1.0,2.9**2,1.0,0.3)    #Setting creator for gap region
gap=A_FMM.layer(NX,0,cr)       #Creating gap layer

#Creating stack
mat=[waveguide,gap,waveguide,gap,waveguide]  #Creating list of layers
d=[0.5,0.15,0.15,0.15,0.5]                   #Creating list of thicknesses
st=A_FMM.stack(mat,d)                        #Creating stack
st.count_interface()                         #Calling count_interface, always do this right after stack creation
st.plot_stack()                              #Plotting stack epsilon

#Adding pml transform
st.transform_complex(ex=0.6)                 #Transform only in x direction. Total thickness of the non trasformed region in 0.6 of the period. In this particular case trasformation start 150 nmm above and under the waveguide

#Doing the actual calculation
st.solve(k0)            #Calculating scattering matrix of entire structure
R1=st.get_R(1,1)        #Get reflection of fundamental mode (TE)
T1=st.get_T(1,1)        #Get transmission of fundamental mode (TE)
R2=st.get_R(2,2)        #Get reflection of second mode (TM)
T2=st.get_T(2,2)        #Get transmission of second mode (TM)

#Printing results
print(5*'%15.9f' % (k0,R1,T1,R2,T2))

#Plotting filed, comment in not wanted
st.plot_E(i=1,pdfname='pml_mode1')    #Plotting fundamental mode
st.plot_E(i=2,pdfname='pml_mode2')    #Plotting second mode







