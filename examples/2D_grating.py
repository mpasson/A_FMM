# Calculate reflection and transmission for TE and TM for 1D PhC slab
# Structure as following
#       Vertical cross section                Cross section of pattern
#                                          ______________________________
#  |                              |       |                              |     Data   
#  |        Silicon Oxide         |       |           Silicon            |     Period:              650 nm
#  |        (epsilon=2.0)         |       |        ______________        |     Si Thickness:        220 nm
#  |        ______________        |       |       |              |       |     Etching Depth:       70 nm
#  |       |              |       |       |       |              |       |     Duty Cycle:          50 %
#  |_______|              |_______|       |       |   Silicon    |       |     Wavelenght Interval: 0.9-1.1 microns
#  |                              |       |       |    Oxide     |       |     Angles Interval:     0-89 degrees
#  |     Silicon (epsilon=12.0)   |       |       |              |       |
#  |______________________________|       |       |              |       |
#  |                              |       |       |______________|       |
#  |        Silicon Oxide         |       |                              |
#  |        (epsilon=2.0)         |       |           Silicon            | 
#  |                              |       |______________________________|



#Allow import of A_FMM from repository folder, if A_FMM is already in path can be removed
import sys
sys.path.append('../.')
#Other imports
import numpy as np
import A_FMM

#Initializing creator structure for layer creation
cr=A_FMM.creator()

#Computational cell for the layer is assumed 1D with period of 650 nanometers, only NX is set
NX,NY=3,3
theta_l=np.linspace(0.0,89.0,90)    # Setting vector of angles
k0_l=np.linspace(0.9,1.1,201)       # Setting vector of wavelenght

#Creting layers involved in structure
cr.slab(2.0,2.0,2.0,0.5)       #Setting creator for Silicon Oxide
SiO=A_FMM.layer(NX,NY,cr)       #Creating Silicon Oxide
cr.rect(12.0,12.0,0.5,0.5)     #Setting creator for Silicon
Si=A_FMM.layer(NX,NY,cr)        #Creating Silicon
cr.rect(2.0,12.0,0.5,0.5)      #Setting creator for patterned region
Pat=A_FMM.layer(NX,NY,cr)       #Creating patterned region


#Creating stack
mat=[SiO,Pat,Si,SiO]                         #Creating list of layers
d=[0.1,0.107,0.23,0.1]                       #Creating list of thicknesses
st=A_FMM.stack(mat,d)                        #Creating stack
st.count_interface()                         #Calling count_interface, always do this right after stack creation
st.plot_stack()                              #Plotting stack epsilon
Pat.eps_plot('Pat')                          #Plotting horizontal cross-section of patterned region

#Possible printing of G to see diffraction orders
#for g in SiO.G:
#    print g,SiO.G[g][0],SiO.G[g][1]

for l in k0_l:
    for theta in theta_l:
        k0=0.65/l                                                          # Setting k0 as P/lambda
        st.solve(k0,kx=np.sqrt(2.0)*k0*np.sin(np.pi/180.0*theta))          # Solving system, kx is set as the parallel component of the wavevector (kx=k0*n*sin(theta))
        TM=st.get_T(st.NPW/2,st.NPW/2,ordered='no')                        # Extracting reflection and transmission for TM mode (pol along x)
        RM=st.get_R(st.NPW/2,st.NPW/2,ordered='no')                        # When dealing with uniform media the fundamental mode with pol along x is the NPW/2^th in unordered configuration
        TE=st.get_T(3*st.NPW/2,3*st.NPW/2,ordered='no')                    # Extracting reflection and transmission for TE mode (pol along y)
        RE=st.get_R(3*st.NPW/2,3*st.NPW/2,ordered='no')                    # When dealing with uniform media the fundamental mode with pol along y is the 3*NPW/2^th in unordered configuration
        print 6*'%15.8f' % (l,theta,TE,RE,TM,RM)                           # Printing results
    print ''                                                               # Inserting blank line for easy map plot with gnuplot




