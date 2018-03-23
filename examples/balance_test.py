import sys
sys.path.append('../.')
#Other imports
import numpy as np
import A_FMM


AIR=A_FMM.layer_uniform(0,0,1.0)
TEST=A_FMM.layer_uniform(0,0,2.0+0.1j)

mat=[AIR,TEST,AIR]
d=[0.0,0.2,0.0]

st=A_FMM.stack(mat,d)
st.count_interface()

for l in np.linspace(0.1,0.9,500):
    st.solve(1.0/l)
    RE=st.get_R(0,0,ordered='no')
    RM=st.get_R(1,1,ordered='no')
    TE=st.get_T(0,0,ordered='no')
    TM=st.get_T(1,1,ordered='no')
    u=[1.0,0.0]
    [P1E,P2E,AE]=st.get_energybalance(u)
    u=[0.0,1.0]
    [P1M,P2M,AM]=st.get_energybalance(u)
    print 12*'%8.4f' % (l,0,RE,TE,P1E,P2E,AE,RM,TM,P1M,P2M,AM)
