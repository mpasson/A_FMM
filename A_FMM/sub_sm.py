import numpy as np
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt
import numpy.linalg
import inspect
from matplotlib.backends.backend_pdf import PdfPages


def savefig(pdf, fig):
    if isinstance(pdf,PdfPages):
        pdf.savefig(fig)
    elif isinstance(pdf, str):
        with PdfPages(pdf) as a:
            a.savefig(fig)
    if pdf is not None: plt.close()

def get_user_attributes(cls):
    boring = dir(type('dummy', (object,), {}))
    return [item
            for item in inspect.getmembers(cls)
            if item[0] not in boring]

def createG(Ng1,Ng2):
    dic={}
    n=0
    for i in range(-Ng1,Ng1+1):
        for j in range(-Ng2,Ng2+1):
#    for j in range(-Ng2,Ng2+1):
#        for i in range(-Ng1,Ng1+1):
            dic[n]=(i,j)
            n+=1
    return dic



def inter_(n,x_list, eps_list):
    if (n==0):
        return eps_list[0] + sum([-(eps_list[(j+1) % len(eps_list)]-eps_list[j])*x_list[j] for j in range(len(x_list))])
    else:
        return sum([(eps_list[(j+1) % len(eps_list)]-eps_list[j])*np.exp(-(0+2j)*np.pi*n*x_list[j]) for j in range(len(x_list))])/((0+2j)*np.pi*n)

def inter_v(n,x_list, eps_list):
    if (n==0):
        return 1.0/eps_list[0] + sum([-(1.0/eps_list[(j+1) % len(eps_list)]-1.0/eps_list[j])*x_list[j] for j in range(len(x_list))])
    else:
        return sum([(1.0/eps_list[(j+1) % len(eps_list)]-1.0/eps_list[j])*np.exp(-(0+2j)*np.pi*n*x_list[j]) for j in range(len(x_list))])/((0+2j)*np.pi*n)

def fou(nx,ny,x_list,y_list,eps_lists):
    N=len(x_list)
    f=[]
    for i in range(N):
        f.append(inter_(ny,y_list,eps_lists[i]))
    return (1+0j)*inter_(nx,x_list,f)


def create_epsilon(G,x_list,y_list,eps_lists):
    D=len(G)
    F=np.zeros((D,D),complex)
    for i in range(D):
        for j in range(D):
            F[i,j]=fou(G[i][0]-G[j][0],G[i][1]-G[j][1],x_list,y_list,eps_lists)
    return F    

def fou_v(nx,ny,x_list,y_list,eps_lists):
    N=len(x_list)
    f=[]
    for i in range(N):
        f.append(inter_v(ny,y_list,eps_lists[i]))
    return (1+0j)*inter_(nx,x_list,f)


def fou_yx(Nx,Ny,G,x_list,y_list,eps_lists):
    f=[]
    D=len(G)
    nx=len(x_list)
    for i in range(nx):
        f.append(linalg.inv(linalg.toeplitz([inter_v(j+Ny,y_list,eps_lists[i]) for j in range(-Ny,Ny+1)])))
    F=np.zeros((D,D),complex)
    for i in range(D):
        for j in range(D):
            F[i,j]=inter_(G[i][0]-G[j][0],x_list,f)[+G[i][1]+Ny,+G[j][1]+Ny]
    return F


def fou_xy(Nx,Ny,G,x_list,y_list,eps_lists):
    f=[]
    D=len(G)
    nx=len(x_list)
    ny=len(y_list)
    for i in range(ny):
        eps_t=[eps_lists[j][i] for j in range(nx)]                
        f.append(linalg.inv(linalg.toeplitz([inter_v(j+Nx,x_list,eps_t) for j in range(-Nx,Nx+1)])))
    F=np.zeros((D,D),complex)
    for i in range(D):
        for j in range(D):
            F[i,j]=inter_(G[i][1]-G[j][1],y_list,f)[G[i][0]+Nx,G[j][0]+Nx]
    return F


def plot(n_max,x_list,eps_list,N=100):
    x=np.linspace(-0.5,0.5,N)
    y=np.zeros(N,complex)
    for n in range(-n_max,n_max+1):
        y+= inter_(n,x_list,eps_list)*np.exp((0+2j)*np.pi*n*x)
#        y+= slab_f(n,0.4)*np.exp((0+2j)*np.pi*n*x)

    plt.plot(x,np.real(y))
    plt.show()


def createK(dic,k0,kx=0.0,ky=0.0,Nyx=1.0):
    D=len(dic)
    K1=np.zeros((D,D),dtype=complex)
    K2=np.zeros((D,D),dtype=complex)
    for i in range(D):
#        K1[i,i]=2.0*np.pi*dic[i][0]/k0*(1+0j)
#        K2[i,i]=2.0*np.pi*dic[i][1]/k0*(1+0j)

        K1[i,i]=(1+0j)*(dic[i][0]+kx)/k0
        K2[i,i]=(1+0j)*(dic[i][1]+ky)/k0/Nyx
    return (K1,K2)


def create_2order(dic,K1,K2,INV,EPS1,EPS2):
    D=len(dic)
    ID=np.identity(D,dtype='complex')
    I11=-np.dot(K2,np.dot(INV,K1))
    I12=np.dot(K2,np.dot(INV,K2)) - ID
    I21=ID - np.dot(K1,np.dot(INV,K1))
    I22=np.dot(K1,np.dot(INV,K2))
    B12=np.vstack([np.hstack([I11,I12]),np.hstack([I21,I22])])
    I11=np.dot(K1,K2)
    I22=-np.dot(K1,K2)
    I21=np.dot(K1,K1)-EPS2
    I12=EPS1-np.dot(K2,K2)
    B21=np.vstack([np.hstack([I11,I12]),np.hstack([I21,I22])])
#    B=np.dot(B12,B21)
#    print 3*'%15.8e' % (numpy.linalg.cond(B12),numpy.linalg.cond(B21),numpy.linalg.cond(B))
    return np.dot(B12,B21)

#convention of Li article
def create_2order_new(D,Kx,Ky,INV,EPS1,EPS2):
    ID=np.identity(D,dtype='complex')
    F=np.vstack([np.hstack([np.dot(np.dot(Kx,INV),Ky),ID-np.dot(np.dot(Kx,INV),Kx)]),np.hstack([np.dot(np.dot(Ky,INV),Ky)-ID,-np.dot(np.dot(Ky,INV),Kx)])])
    G=np.vstack([np.hstack([-np.dot(Kx,Ky),np.dot(Kx,Kx)-EPS2]),np.hstack([EPS1-np.dot(Ky,Ky),np.dot(Kx,Ky)])])
    #old version for control
    #G=np.vstack([np.hstack([-np.dot(Kx,Ky),np.dot(Kx,Kx)-EPS2]),np.hstack([EPS1-np.dot(Ky,Ky),np.dot(Ky,Kx)])])
    #F=np.vstack([np.hstack([np.dot(np.dot(Kx,INV),Ky),ID-np.dot(np.dot(Kx,INV),Kx)]),np.hstack([np.dot(np.dot(Ky,INV),Ky)-ID,-np.dot(np.dot(Ky,INV),Kx)])])
    return G,np.dot(F,G)


#convention of Lalanne article
#def create_2order_new(D,Kx,Ky,INV,EPS1,EPS2):
#    ID=np.identity(D,dtype='complex')
#    F=np.vstack([np.hstack([np.dot(np.dot(Ky,INV),Kx),ID-np.dot(np.dot(Ky,INV),Ky)]),np.hstack([np.dot(np.dot(Kx,INV),Kx)-ID,-np.dot(np.dot(Kx,INV),Ky)])])
#    G=np.vstack([np.hstack([np.dot(Kx,Ky),EPS1-np.dot(Ky,Ky)]),np.hstack([np.dot(Kx,Kx)-EPS2,-np.dot(Kx,Ky)])])
#    return np.dot(F,G)



def fou_t(n,e):
    q=1.0-e
    return 1.0*(n==0)-0.5*q*(-1)**n*(np.sinc(n*q)+0.5*np.sinc(n*q-1)+0.5*np.sinc(n*q+1))

def t_inv(x,e):
    q=1.0-e
    e=0.5*e
    xr=np.copy(x)
    ind=np.abs(x)>e
    xr[ind]=np.sign(x[ind])*(e+q/np.pi*np.arctan(np.pi/q*(np.abs(x[ind])-e)))
    return xr


def t_dir(x,e):
    q=1.0-e
    e=0.5*e
    xr=np.copy(x)
    ind=np.abs(x)>e
    xr[ind]=np.sign(x[ind])*(e+q/np.pi*np.tan(np.pi/q*(np.abs(x[ind])-e)))
    return xr



def fou_complex_t(n,e,g):
    q=1.0-e
    return 1.0*(n==0)-0.5*q*(-1)**n*((1.0+0.25*g)*np.sinc(n*q)+0.5*np.sinc(n*q-1)+0.5*np.sinc(n*q+1)-g/8.0*(np.sinc(n*q-2)+np.sinc(n*q+2)))


def num_fou(func,args,G,NX,NY,Nyx):
    #[X,Y]=np.meshgrid(np.linspace(-0.5,0.5,NX),np.linspace(-0.5,0.5,NY))
    [Y,X]=np.meshgrid(np.linspace(-0.5,0.5,NY),np.linspace(-0.5,0.5,NX))
    F=func(X,Y/Nyx,*args)
    F=np.fft.fftshift(F)
    FOU=np.fft.fft2(F)/NX/NY
    EPS=np.zeros((len(G),len(G)),dtype=complex)
    for i in range(len(G)):
        for j in range(len(G)):
            EPS[i,j]=FOU[G[i][0]-G[j][0],G[i][1]-G[j][1]]
    return EPS

def num_fou_xy(func,args,nx,ny,G,NX,NY,Nyx):
    #[X,Y]=np.meshgrid(np.linspace(-0.5,0.5,NX),np.linspace(-0.5,0.5,NY))
    [Y,X]=np.meshgrid(np.linspace(-0.5,0.5,NY),np.linspace(-0.5,0.5,NX))
    F=1.0/func(X,Y/Nyx,*args)
    F=np.fft.fftshift(F)
    np.shape(F)
    FOU=np.fft.fft(F,axis=0)/NX
    #plt.figure()
    #plt.imshow(np.abs(F),origin='lower')
    #plt.colorbar()
    #plt.savefig('F.png')
    #plt.figure()
    #plt.imshow(np.abs(FOU[:,:20]),aspect='auto',origin='lower')
    #plt.colorbar()
    #plt.savefig('FOU.png')
    TEMP1=np.zeros((NY,2*nx+1,2*nx+1),dtype=complex)
    for i in range(-nx,nx+1):
        for j in range(-nx,nx+1):
            TEMP1[:,i,j]=FOU[i-j,:]
    #print TEMP1[900,:,:]
    TEMP2=np.linalg.inv(TEMP1)
    TEMP3=np.fft.fft(TEMP2,axis=0)/NY
    EPS=np.zeros((len(G),len(G)),dtype=complex)
    for i in range(len(G)):
        for j in range(len(G)):
            EPS[i,j]=TEMP3[G[i][1]-G[j][1],G[i][0],G[j][0]]
    return EPS
    #return None


def num_fou_yx(func,args,nx,ny,G,NX,NY,Nyx):
    #[X,Y]=np.meshgrid(np.linspace(-0.5,0.5,NX),np.linspace(-0.5,0.5,NY))
    [Y,X]=np.meshgrid(np.linspace(-0.5,0.5,NY),np.linspace(-0.5,0.5,NX))
    F=1.0/func(X,Y/Nyx,*args)
    F=np.fft.fftshift(F)
    FOU=np.fft.fft(F,axis=1)/NY
    #plt.figure()
    #plt.imshow(np.abs(F),origin='lower')
    #plt.colorbar()
    #plt.savefig('F.png')
    #plt.figure()
    #plt.imshow(np.abs(FOU[:,:20]),aspect='auto',origin='lower')
    #plt.colorbar()
    #plt.savefig('FOU.png')
    TEMP1=np.zeros((NX,2*ny+1,2*ny+1),dtype=complex)
    for i in range(-ny,ny+1):
        for j in range(-ny,ny+1):
            TEMP1[:,i,j]=FOU[:,i-j]
    #print TEMP1[900,:,:]
    TEMP2=np.linalg.inv(TEMP1)
    TEMP3=np.fft.fft(TEMP2,axis=0)/NX
    EPS=np.zeros((len(G),len(G)),dtype=complex)
    for i in range(len(G)):
        for j in range(len(G)):
            EPS[i,j]=TEMP3[G[i][0]-G[j][0],G[i][1],G[j][1]]
    return EPS
    #return None





