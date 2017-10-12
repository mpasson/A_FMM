# Calculate reflection and transmission for TE and TM for 1D PhC slab
# Structure as following
#  |                              |       Data   
#  |        Silicon Oxide         |       Period:              650 nm
#  |        (epsilon=2.0)         |       Si Thickness:        220 nm
#  |          __________          |       Etching Depth:       70 nm
#  |         |          |         |       Duty Cycle:          50 %
#  |_________|          |_________|       Wavelenght Interval: 0.9-1.1 microns
#  |                              |       Angles Interval:     0-89 degrees
#  |     Silicon (epsilon=12.0)   |
#  |______________________________|
#  |                              |
#  |        Silicon Oxide         |
#  |        (epsilon=2.0)         |
#  |                              |



#Allow import of A_FMM from repository folder, if A_FMM is already in path can be removed
import sys
sys.path.append('../.')
#Other imports
import numpy as np
import A_FMM

#Initializing creator structure for layer creation
cr=A_FMM.creator()

#Computational cell for the layer is assumed 1D with period of 650 nanometers, only NX is set
NX=20
k0=0.65/1.55 #Setting energy (in unit of period/lambda) (lambda_0=1.55 microns)

#Creting layers involved in structure
cr.slab(2.0,2.0,2.0,0.5)       #Setting creator for Silicon Oxide
SiO=A_FMM.layer(NX,0,cr)       #Creating Silicon Oxide
cr.slab(12.0,12.0,12.0,0.5)    #Setting creator for Silicon
Si=A_FMM.layer(NX,0,cr)        #Creating Silicon
cr.slab(12.0,2.0,2.0,0.5)      #Setting creator for patterned region
Pat=A_FMM.layer(NX,0,cr)       #Creating patterned region


#Creating stack
mat=[SiO,Pat,Si,SiO]                         #Creating list of layers
d=[0.1,0.107,0.23,0.1]                   #Creating list of thicknesses
st=A_FMM.stack(mat,d)                        #Creating stack
st.count_interface()                         #Calling count_interface, always do this right after stack creation
st.plot_stack()                              #Plotting stack epsilon



theta_l=np.linspace(0.0,89.0,90)    # Setting vector of angles
k0_l=np.linspace(0.9,1.1,201)       # Setting vector of wavelenght

#for theta in theta_l:
for l in k0_l:
    for theta in theta_l:
        k0=0.65/l                                                          # Setting k0 as P/lambda
        st.solve(k0,kx=np.sqrt(2.0)*k0*np.sin(np.pi/180.0*theta))          # Solving system, kx is set as the parallel component of the wavevector (kx=k0*n*sin(theta))
        TM=st.get_T(NX+1,NX+1,ordered='no')                                # Extracting reflection and transmission for TM mode (pol along x)
        RM=st.get_R(NX+1,NX+1,ordered='no')                                # When dealing with uniform media the fundamental mode with pol along x is the NPW+1^th in unordered configuration
        TE=st.get_T(3*NX+2,3*NX+2,ordered='no')                            # Extracting reflection and transmission for TE mode (pol along y)
        RE=st.get_R(3*NX+2,3*NX+2,ordered='no')                            # When dealing with uniform media the fundamental mode with pol along y is the 3*NPW+2^th in unordered configuration
        print 6*'%15.8f' % (l,theta,TE,RE,TM,RM)                           # Printing results
    print ''                                                               # Inserting blank line for easy map plot with gnuplot




