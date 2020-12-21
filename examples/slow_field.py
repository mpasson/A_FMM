#Allow import of A_FMM from repository folder, if A_FMM is already in path can be removed
import sys
sys.path.append('../.')
#Other imports
import numpy as np
import A_FMM
import copy
from matplotlib.backends.backend_pdf import PdfPages

def proc(k0):                                          # function which calculates the bands for the given k0
    st.solve(k0)                                       #Create the scattering matirx of structure
    st.bloch_modes()                                   #Solve for Bloch modes
    BK=st.Bk                                           #saving Bloch vectors
    return BK    

#Computational cell for the layer is assumed 1D, only NX is set. The supercell dimension is set by the s parameter. This gives the scale for wavevectors, since then coordinate transfomation is applied.

#SOI 310 nm
NX=11                              # Setting X truncation order
NY=5                               # Setting Y truncation order
k0=1.0/1.3                          # Setting array of wavevectors
#k0_ll=np.linspace(0.6,0.64,24)       # Setting array of wavevectors
SOI=0.300                           # thickness in micron
t=0.0                              # cladding thickness in micron
W1=0.0                              #W1 in microns
W2=0.4                              #W1 in microns
FF=0.5                              #Filling fraction
P=0.240                             #Period in microns
ax=1.0                              #ax  in micron
ay=0.75                              #ay  in micron
ratio=ay/ax                         #ay/ax 
eps_Si=12.299
eps_SiO2=2.09
ex=0.8                               #parameter for the x coordinate transform
ey=0.7                               #parameter for the y coordinate transform
lam_t=1.31                            #target wavelength for tuning of the band edge
k0=k0*ax                       # setting right units for wavevectors


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


#Initializing creator structure for layer creation
cr=A_FMM.creator()
#Creting layers involved in structure
cr.ridge(eps_Si,eps_SiO2,eps_SiO2,W1/ax,SOI/ay,t/ay,x_offset=0.0,y_offset=0.0)          #Creator for thin part
narrow=A_FMM.layer(NX,NY,cr,Nyx=ratio)                                                    #Creating thin part
cr.ridge(eps_Si,eps_SiO2,eps_SiO2,W2/ax,SOI/ay,t/ay,x_offset=0.0,y_offset=0.0)          #Creator for thick part
wide=A_FMM.layer(NX,NY,cr,Nyx=ratio)                                                   #Creating thick part

mat=[narrow,narrow,wide,narrow]                          #creating list of layer
narrow.eps_plot('narrow')                                  #plotting eps of layer
wide.eps_plot('wide')

pdf=PdfPages('Bloch_TE.pdf')
pdf2=PdfPages('Bloch_TM.pdf')
pdf3=PdfPages('EPS.pdf')


d=[0.0,FF*P/ax,(1.0-FF)*P/ax,0.0]                        #creating list of thicknesses
st=A_FMM.stack(mat,d)                                    #Creating stack
st.count_interface()                                     #Calling count_interface, always do this right after stack creation
st.transform(ex=ex,ey=ey)                              #Setting the coordinate transform (in unit of cell)


for gap in np.linspace(0.00,0.1,11):

    st.d=[0.0,gap/ax,0.12/ax,0.0]                        #creating list of thicknesses

    st.plot_stack(pdf=pdf3)
    st.solve(k0)                                           #Create the scattering matirx of structure
    st.bloch_modes()                                        #Solve for Bloch modes
    BK=st.Bk                                                #saving Bloch vectors

    K=np.sort(BK)[:-3:-1]*(gap+0.12)/np.pi
    neff=0.5*K*ax/(gap+0.12)/k0

    print(gap,K,st.tot_thick)
    inds=np.argsort(BK.real)
    st.plot_E_periodic(inds[-1],r=5,dz=0.01,pdf=pdf,title=f'TE gap={gap:.4} neff={neff[0]:.4}',func=np.abs)
    st.plot_E_periodic(inds[-2],r=5,dz=0.01,pdf=pdf2,title=f'TM gap={gap:.4} neff={neff[1]:.4}',func=np.abs)

pdf.close()
pdf2.close()
pdf3.close()



