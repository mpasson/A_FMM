#Allow import of A_FMM from repository folder, if A_FMM is already in path can be removed
import sys
sys.path.append('../.')
#Other imports
import numpy as np
import A_FMM
from multiprocessing import Pool
from scipy.optimize import curve_fit
import copy

def proc(k0):                                          # function which calculates the bands for the given k0
    st.solve(k0)                                       #Create the scattering matirx of structure
    st.bloch_modes()                                   #Solve for Bloch modes
    BK=st.Bk                                           #saving Bloch vectors
    return BK    

#Computational cell for the layer is assumed 1D, only NX is set. The supercell dimension is set by the s parameter. This gives the scale for wavevectors, since then coordinate transfomation is applied.

#SOI 310 nm
NX=11                               # Setting X truncation order
NY=5                              # Setting Y truncation order
#k0_l=np.linspace(0.01,1.0,30)      # Setting array of wavevectors
k0_ll=np.linspace(0.6,0.64,81)       # Setting array of wavevectors
SOI=0.220                           # thickness in micron
t=0.100                              # cladding thickness in micron
W1=0.300                              #W1 in microns
W2=0.650                              #W1 in microns
FF=0.5                              #Filling fraction
P=0.310                             #Period in microns
ax=1.5                              #ax  in micron
ay=0.5                              #ay  in micron
ratio=ay/ax                         #ay/ax 
#eps_Si=12.299                       #lam =1.31
eps_Si=12.081                       #lam =1.55
eps_SiO2=2.09
ex=0.8                               #parameter for the x coordinate transform
ey=0.7                               #parameter for the y coordinate transform
lam_t=1.55                            #target wavelength for tuning of the band edge
k0_ll=k0_ll*ax                       # setting right units for wavevectors
NN=5e17
PP=4e17


#SOI 220 nm
#NX=11                               # Setting X truncation order
#NY=5                                # Setting Y truncation order
#k0_l=np.linspace(0.01,1.0,30)      # Setting array of wavevectors
#k0_l=np.linspace(0.5,0.8,16)        # Setting array of wavevectors
#SOI=0.310                           # thickness in micron
#t=0.05                              # cladding thickness in micron
#W1=0.1                              #W1 in microns
#W2=0.8                              #W1 in microns
#FF=0.5                              #Filling fraction
#P=0.145                               #Period in microns
#ax=1.5                              #ax  in micron
#ay=0.5                              #ay  in micron
#ratio=ay/ax                         #ay/ax 
#eps_Si=12.299
#eps_SiO2=2.0
#ex=0.8                               #parameter for the x coordinate transform
#ey=0.7                               #parameter for the y coordinate transform
#lam_t=1.3                            #target wavelength for tuning of the band edge
#k0_ll=k0_ll*ax                       # setting right units for wavevectors

#depsN=2.0*np.sqrt(eps)*(3.64e-10*(k0*1e-6)**2*NN-2.8e-5j*(k0*1e-6)**3*NN)
#depsP=2.0*np.sqrt(eps)*(3.51e-6*(k0*1e-6)**2*PP**0.8-1.91e-5j*(k0*1e-6)**3*PP)

depsN=2.0*np.sqrt(eps_Si)*3.64e-10*(lam_t*1e-6)**2*NN
depsP=2.0*np.sqrt(eps_Si)*3.51e-6*(lam_t*1e-6)**2*PP**0.8

ni=1.4e10
e0=8.85e-12
T=300
kb=1.38e-23
ec=1.6e-19
V0=kb*T/ec*np.log(NN*PP/ni**2)
epsn=eps_Si-depsN
epsp=eps_Si-depsP

V=float(sys.argv[-1])
xp=np.sqrt(2.0*e0*eps_Si*NN/(ec*PP*(NN+PP)*1e6)*(V0+V))
xn=np.sqrt(2.0*e0*eps_Si*PP/(ec*NN*(NN+PP)*1e6)*(V0+V))


#Initializing creator structure for layer creation
cr=A_FMM.creator()
#Creting layers involved in structure
#cr.ridge(eps_Si,eps_SiO2,eps_SiO2,W1/ax,SOI/ay,t/ay,x_offset=0.0,y_offset=0.0)          #Creator for thin part
cr.ridge_pn(eps_Si,epsp,epsn,eps_SiO2,eps_SiO2,W1/ax,SOI/ay,t/ay,-xp/ax*1e6,xn/ax*1e6)
narrow=A_FMM.layer(NX,NY,cr,Nyx=ratio)                                                    #Creating thin part
#cr.ridge(eps_Si,eps_SiO2,eps_SiO2,W2/ax,SOI/ay,t/ay,x_offset=0.0,y_offset=0.0)          #Creator for thick part
cr.ridge_pn(eps_Si,epsp,epsn,eps_SiO2,eps_SiO2,W2/ax,SOI/ay,t/ay,-xp/ax*1e6,xn/ax*1e6)
wide=A_FMM.layer(NX,NY,cr,Nyx=ratio)                                                   #Creating thick part

mat=[narrow,narrow,wide,narrow]                          #creating list of layer
narrow.eps_plot('narrow')                                  #plotting eps of layer
wide.eps_plot('wide')


for i in range(1):                      # no iterations, one single calculation
#for i in range(10):                    # loop for tuning the period of the waveguide
    k0_l=copy.deepcopy(k0_ll)           # Setting array of wavevectors

    d=[0.0,FF*P/ax,(1.0-FF)*P/ax,0.0]                        #creating list of thicknesses
    st=A_FMM.stack(mat,d)                                    #Creating stack
    st.count_interface()                                     #Calling count_interface, always do this right after stack creation
    st.transform(ex=ex,ey=ey)                              #Setting the coordinate transform (in unit of cell)
#    st.plot_stack()                                         #Plotting stack epsilon



    if __name__=='__main__':                               # parallelization
        pool=Pool(4)                                       # does the parallelization with N=4 processors 
        BKar=pool.map(proc,k0_l)                             # calculates bands using the above function
        del pool                                           # removes pool object
  
#    BKar=[]                                                #not parallel syntax of previos lines
#    for k0 in k0_l:
#        st.solve(k0)                                    #Create the scattering matirx of structure
#        st.bloch_modes()                                   #Solve for Bloch modes
#        BK=st.Bk                                           #saving Bloch vectors
#        Bkar.append(BK)	

    k0_l=k0_l/ax
    bkmax=[]
    f=open('bands_%i.out' % (i),'w')
    f2=open('bands_max_%i.out' % (i),'w')
    for k0,BK in zip(k0_l,BKar):                           #double loop: over energies and list of bloch vectors
        f.write('%10.5f' % (k0))                           #writing data to output file
        for k in BK:
            f.write('%10.5f' % (k.real))
        f.write('\n')
        #print 2*'%15.8f' % (k0,max(BK)*st.tot_thick/np.pi)         #writing mode with maximum bloch vector
        bkmax.append(max(BK).real*st.tot_thick/np.pi)
        f2.write('%25.15f %25.15f \n' % (k0,max(BK).real*st.tot_thick/np.pi))   #writing data to output file
    f.close()
    f2.close()
    #possible part for fit
    def func(x,om,n,U):                                    #defining fitting function
        d=(1-x)**2
        return om**2 + (d - np.sqrt(4.0*d+(1-d)**2*U**2))/(4.0*n**2)
    kb=np.array(bkmax)                                     #converting data to array
    A=kb>0.999999                                          #finding band gap limit
    ind=list(A).index(True)                                #keep only the point under the band edge by truncating the arrays
    k0_l=k0_l[:ind]*P
    kb=kb[:ind]
#    for t in zip(k0_l,kb):
#        print 2*'%15.8f' %  t

    om_0=0.17                                              #setting initial guess for fit
    n=3.1
    U=0.01
    p0=[om_0,n,U]
    RES=curve_fit(func,kb,k0_l**2,p0=p0)                   #fitting
    [om_0,n,U]=RES[0]                                      #retrieve results
    om_band=np.sqrt(om_0**2.0-0.25*abs(U)/n**2.0)
    lam_band=P/om_band                                     #wavelength of the band edge
    print 8*'%15.8f' % (i,P,om_0,n,U,om_band,lam_band,om_band*lam_t)
    if abs(lam_t-lam_band)<0.001:
        quit()
    P=P*lam_t/lam_band                                     #defining new period for next tuning step
pass                                                       #ends loop for tuning the period
quit()


