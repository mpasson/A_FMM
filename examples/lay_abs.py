#Allow import of A_FMM from repository folder, if A_FMM is already in path can be removed
import sys
sys.path.append('../.')
#Other imports
import numpy as np
import A_FMM

#Initializing creator structure for layer creation
cr=A_FMM.creator()

#Computational cell for the layer is assumed 1D with period of 1 micron, only NX is set
NX=0
k0=1.0/1.55 #Setting energy (in unit of period/labda) (lambda_0=975 nm)

#Creting layers involved in structure
l1=A_FMM.layer_uniform(NX,0,1.0)
l2=A_FMM.layer_uniform(NX,0,12.0)
l3=A_FMM.layer_uniform(NX,0,1.0)
l4=A_FMM.layer_uniform(NX,0,4.0)

mat=[l1,l2,l1]
d=[0.2,10.0,0.2]
st=A_FMM.stack(mat,d)
st.count_interface()


#for l in np.linspace(1.5,1.6,101):
#    st.solve(1.0/l)
#    print '%10.6f %10.6f %10.6f' %  (l,st.get_T(1,1,ordered='yes'),st.get_R(1,1,ordered='yes'))
#quit()

l=1.54
st.solve(1.0/l)
u=st.create_input({1:1.0})
P=st.get_prop(u,[0,1,2,3])
for p in P:
    print P[p][2]










