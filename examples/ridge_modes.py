#Allow import of A_FMM from repository folder, if A_FMM is already in path can be removed
import sys
sys.path.append('../.')
#Other imports
import numpy as np
import A_FMM
from matplotlib.backends.backend_pdf import PdfPages


#Defining data
Nx=10
Ny=10
k0_l=np.linspace(0.01,2.0,500)


#Defining creator
cr=A_FMM.creator()
cr.ridge(12.0,2.0,2.0,0.4,0.31,0.15)
ridge=A_FMM.layer(Nx,Ny,cr)
ridge.trasform(ex=0.8,ey=0.8)
ridge.eps_plot('ridge')


for k0 in k0_l:
    ridge.mode(k0)
    neff=np.sort(ridge.gamma)[-10:]
    print 11*'%15.8f' % ((k0,)+tuple(neff))



k0=0.2    
ridge.mode(k0)
a=PdfPages('k='+str(k0)+'.pdf')
for i in range(1,11):
    ridge.plot_field(a,i)
a.close()


k0=0.55    
ridge.mode(k0)
a=PdfPages('k='+str(k0)+'.pdf')
for i in range(1,11):
    ridge.plot_field(a,i)
a.close()

k0=0.645    
ridge.mode(k0)
a=PdfPages('LAM155.pdf')
for i in range(1,11):
    ridge.plot_field(a,i)
a.close()

k0=0.7634    
ridge.mode(k0)
a=PdfPages('LAM131.pdf')
for i in range(1,11):
    ridge.plot_field(a,i)
a.close()

k0=1.0
ridge.mode(k0)
a=PdfPages('k='+str(k0)+'.pdf')
for i in range(1,11):
    ridge.plot_field(a,i)
a.close()

k0=1.8
ridge.mode(k0)
a=PdfPages('k='+str(k0)+'.pdf')
for i in range(1,11):
    ridge.plot_field(a,i)
a.close()
