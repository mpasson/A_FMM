# Calculate reflection and transmission for TE and TM for 1D PhC slab
# Calculation is done only at normal incidence, but all diffraction orders are calculated
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
NX,NY=4,4
k0=0.65/1.55 #Setting energy (in unit of period/lambda) (lambda_0=1.55 microns)

#Creating variables for diffractions order
NPW=(2*NX+1)*(2*NY+1)
TEord=np.zeros(NPW,dtype=float)
REord=np.zeros(NPW,dtype=float)
TMord=np.zeros(NPW,dtype=float)
RMord=np.zeros(NPW,dtype=float)

#Creting layers involved in structure
cr.slab(2.0,2.0,2.0,0.5)       #Setting creator for Silicon Oxide
SiO=A_FMM.layer(NX,NY,cr)       #Creating Silicon Oxide
cr.slab(12.0,12.0,12.0,0.5)    #Setting creator for Silicon
Si=A_FMM.layer(NX,NY,cr)        #Creating Silicon
cr.rect(2.0,12.0,0.5,0.5)      #Setting creator for patterned region
Pat=A_FMM.layer(NX,NY,cr)       #Creating patterned region


#Creating stack
mat=[SiO,Pat,Si,SiO]                         #Creating list of layers
d=[0.1,0.107,0.23,0.1]                       #Creating list of thicknesses
st=A_FMM.stack(mat,d)                        #Creating stack
st.count_interface()                         #Calling count_interface, always do this right after stack creation
st.plot_stack()                              #Plotting stack epsilon

NN=951

theta_l=np.linspace(0.0,85.0,1)    # Setting vector of angles
k0_l=np.linspace(0.050,1.0,NN)       # Setting vector of wavelenght

TE_mem,RE_mem,TM_mem,RM_mem=[],[],[],[]



#Possible printing of G to see diffraction orders
#for g in SiO.G:
#    print g,SiO.G[g][0],SiO.G[g][1]



#for theta in theta_l:
for l in k0_l:
    k0=0.65/l                                                              # Setting k0 as P/lambda
    try:                                                                   # Try for avoid numerical problem
        st.solve(k0)                                                       # Solving system, normal incidence
        TM=st.get_T(st.NPW/2,st.NPW/2,ordered='no')                        # Extracting reflection and transmission for TM mode (pol along x)
        RM=st.get_R(st.NPW/2,st.NPW/2,ordered='no')                        # When dealing with uniform media the fundamental mode with pol along x is the NPW/2^th in unordered configuration
        TE=st.get_T(3*st.NPW/2,3*st.NPW/2,ordered='no')                    # Extracting reflection and transmission for TE mode (pol along y)
        RE=st.get_R(3*st.NPW/2,3*st.NPW/2,ordered='no')                    # When dealing with uniform media the fundamental mode with pol along y is the 3*NPW/2^th in unordered configuration
        for k in range(len(st.G)):                                         # Loop for extracting the diffraction orders
            TMord[k]=st.get_T(st.NPW/2,k,ordered='no')                     # TM block down
            RMord[k]=st.get_R(st.NPW/2,k,ordered='no')                     # TM block up
            TEord[k]=st.get_T(st.NPW/2+st.NPW,k+st.NPW,ordered='no')       # TE block down
            REord[k]=st.get_R(st.NPW/2+st.NPW,k+st.NPW,ordered='no')       # TE block up
    except ValueError:                                                     # If numerical problem in met, set all outputs to 0
        TE,RM,TE,RE=0.0,0.0,0.0,0.0
        TEord=np.zeros(NPW,dtype=float)
        REord=np.zeros(NPW,dtype=float)
        TMord=np.zeros(NPW,dtype=float)
        RMord=np.zeros(NPW,dtype=float)    
    TE_up=np.sum(REord)                                                # Summing up all diffraction order
    TE_down=np.sum(TEord)                                              # Summing up all diffraction order
    TM_up=np.sum(RMord)                                                # Summing up all diffraction order
    TM_down=np.sum(TMord)                                              # Summing up all diffraction order
    print 9*'%15.8f' % (l,TE,RE,TE_down,TE_up,TM,RM,TM_down,TM_up)     # Printing results to screen                                                               
    TE_mem.append(np.hstack([l,TEord]))                                # Append diffraction orders for storage
    RE_mem.append(np.hstack([l,REord]))                                # Append diffraction orders for storage
    TM_mem.append(np.hstack([l,TMord]))                                # Append diffraction orders for storage
    RM_mem.append(np.hstack([l,RMord]))                                # Append diffraction orders for storage


TE_mem=np.array(TE_mem)                                                # Converting to array for saving
TM_mem=np.array(TM_mem)                                                # Converting to array for saving
RE_mem=np.array(RE_mem)                                                # Converting to array for saving
RM_mem=np.array(RM_mem)                                                # Converting to array for saving

TITLE='Lambda      '                                                   #Creating title for diffraction ordere file
for g in SiO.G:
    TITLE+='      %+04i,%+04i' % (SiO.G[g][0],SiO.G[g][1])

#Saving diffrection orders
np.savetxt('TEup.out',RE_mem,fmt='%.6e  ',header=TITLE,comments='#')
np.savetxt('TEdown.out',TE_mem,fmt='%.6e  ',header=TITLE,comments='#')
np.savetxt('TMup.out',RM_mem,fmt='%.6e  ',header=TITLE,comments='#')
np.savetxt('TMdown.out',TM_mem,fmt='%.6e  ',header=TITLE,comments='#')



