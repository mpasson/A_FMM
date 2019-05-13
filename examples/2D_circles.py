#Allow import of A_FMM from repository folder, if A_FMM is already in path can be removed
import sys
sys.path.append('../.')
#Other imports
import numpy as np
import A_FMM

#defining arbitrary function for setting the epsilon (in this case circles)
def func(X,Y,r,epsin,epsout):
    ind=np.sqrt(X**2+Y**2)<r
    return epsout+(epsin-epsout)*ind


#setting fixed parameters
a=1.0                    #period in microns
r=0.2                    #radius in microns
t=0.1                    #slab thickness  

#setting computational parameters
N=10                                #truncatuion order (both x and y)
#lam_l=np.linspace(0.5,2.0,151)     #set vector of wavelengths
k0_l=np.linspace(0.2,2.0,181)      #set vector of wavelengths
NPW=(2*N+1)**2



#definig structure
oxide=A_FMM.layer_uniform(N,N,2.0001)                  #defining uniform layer 
slab=A_FMM.layer_num(N,N,func,args=(r,2.0001,12.0001))    #defining arbitrary patterned layer
slab.eps_plot('slab')                               #plotting cross section of slab layer

mat=[oxide,slab,oxide]
d=[0.1,t,0.1]
st=A_FMM.stack(mat,d)
st.count_interface()


for k0 in k0_l:
    st.solve(k0)
    TM=st.get_T(st.NPW/2,st.NPW/2,ordered='no')                        # Extracting reflection and transmission for TM mode (pol along x)
    RM=st.get_R(st.NPW/2,st.NPW/2,ordered='no')                        # When dealing with uniform media the fundamental mode with pol along x is the NPW/2^th in unordered configuration
    TE=st.get_T(3*st.NPW/2,3*st.NPW/2,ordered='no')                    # Extracting reflection and transmission for TE mode (pol along y)
    RE=st.get_R(3*st.NPW/2,3*st.NPW/2,ordered='no')                    # When dealing with uniform media the fundamental mode with pol along y is the 3*NPW/2^th in unordered configuration
    print 5*'%15.8f' % (a/k0,TE,RE,TM,RM)                           # Printing results

