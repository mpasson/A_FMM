#Allow import of A_FMM from repository folder, if A_FMM is already in path can be removed
import sys
sys.path.append('../.')
#Other imports
import numpy as np
import A_FMM
from scipy.optimize import curve_fit
from multiprocessing import Pool

#def func for multiprocessing
def proc(k0):
    st.solve(k0)                                       #Create the scattering matirx of structure
    st.bloch_modes()                                   #Solve for bloch modes
    BK=st.Bk                                           #saving Block vectors
    return BK    


def func(x,om,n,U):                                    #defining fitting function
    d=(1-x)**2
    return om**2 + (d - np.sqrt(4.0*d+(1-d)**2*U**2))/(4.0*n**2)


#Initializing creator structure for layer creation
cr=A_FMM.creator()

#Computational cell for the layer is assumed 1D, only NX is set. The supercell dimension is assumed to be 1 micron. This hive the scale for wavevectors, since then coordinate transfomation is applied.
NX=20                               # Setting X truncation order
#k0_l=np.linspace(0.01,1.0,500)      # Setting array of wavvevectors
k0_l=np.linspace(0.25,0.35,10)     # Setting array of wavvevectors
SOI=0.3                             # thickness in microns
Etch=0.15                           # Etch in microns
FF=0.5                              # Filling fraction
P=0.65                              # Period in microns
eps_core=12.0                       #setting epsilon of core
eps_clad=2.0                        #setting epsilon of cladding


#Creting layers involved in structure
cr.slab(eps_core,eps_clad,eps_clad,SOI)                               #Creator for slab part
slab=A_FMM.layer(NX,0,cr)                                             #Creating slab part
cr.slab(eps_core,eps_clad,eps_clad,SOI-Etch,offset=-0.5*Etch)         #Creator for etched part
etch=A_FMM.layer(NX,0,cr)                                             #Creating etched part


mat=[slab,etch,slab,slab]                              #creating list of layer
d=[0.0,FF*P,(1-FF)*P,0.0]                              #creating list of thicknesses
st=A_FMM.stack(mat,d)                                  #Creating stack
st.count_interface()                                   #Calling count_interface, always do this right after stack creation
st.transform(ex=0.8)                                   #Setting the coordinate transform (in unit of cell)
st.plot_stack()                                        #Plotting stack epsilon

if __name__=='__main__':                               #continue only if program is main (not sure if needed on linux, in windows seems to be)
    p=Pool(4)                                          #defining pool object from multiprocessor module. Number is number of process to be used
    kb=p.map(proc,k0_l)                                #evaluatin the function with parallel map
    
    bkm=[]                                             #defining list for storing max bloch vector
    fre=open('bandsre.out','w')                        #opening file for writing bands
    fim=open('bandsim.out','w')            
    for (k0,BK) in zip(k0_l,kb):                       #writing bands
        fre.write('%15.6f' % (k0))
        fim.write('%15.6f' % (k0))
        bkm.append(max(BK*st.tot_thick/np.pi).real)    #append max bloch vector to list (normalized)
        for kk in BK:
            fre.write('%15.6f' % (kk.real))
            fim.write('%15.6f' % (kk.imag))
        fre.write('\n')
        fim.write('\n')
    fre.close()                                        #closing files for bands
    fim.close() 


    #quit()
    #possible part for fit

    bkm=np.array(bkm)                                        #converting data to array
    A=bkm>0.999999                                          #finding band gap limit
    ind=list(A).index(True)                                #keep only the point under the band edge by truncating the arrays
    k0_l=k0_l[:ind]
    bkm=bkm[:ind]
    om_0=0.17                                              #setting initial guess for fit
    n=3.1
    U=0.01
    p0=[om_0,n,U]
    RES=curve_fit(func,bkm,k0_l**2,p0=p0)                   #fitting
    [om_0,n,U]=RES[0]                                      #retrieve results
    om_lim=np.sqrt(om_0**2.0-0.25*abs(U)/n**2.0)
    for (k0,kk) in zip(k0_l,bkm)
    print 7*'%15.8f' % (P,om_0,n,U,om_lim,P/om_lim,om_lim*1.3)








