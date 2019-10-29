import numpy as np


def conv2(x,y,x0,z0,theta):
    #theta=np.pi*theta/180.0
    #z=-x*np.sin(theta)+z0*np.cos(theta)
    z=(x0-x)*np.sin(theta)+z0*np.cos(theta)
    #r=np.abs(x+z0*np.tan(theta))*np.cos(theta)
    r=np.abs(x-x0+z0*np.tan(theta))*np.cos(theta)
    r=np.sqrt(r**2.0+np.cos(theta)**2.0*y**2.0)
    return (z,r)

def FULL_CONV(X,Y,x0,y0,z0,thetax,thetay):
    if thetax==0.0:
        if thetay==0.0:
            phi=0.0
            theta=0.0
        else:
            #phi=np.arctan(np.tan(np.pi/180.0*thetay)/np.tan(np.pi/180.0*thetax))
            #theta=np.arctan(np.tan(np.pi/180.0*thetay)/np.sin(np.pi/180.0*phi))
            phi=np.pi/2.0
            theta=np.pi/180.0*thetay
    else:
        phi=np.arctan(np.tan(np.pi/180.0*thetay)/np.tan(np.pi/180.0*thetax))
        theta=np.arctan(np.tan(np.pi/180.0*thetax)/np.cos(np.pi/180.0*phi))

    #phi=0.5*np.pi-phi

    Xp=np.cos(phi)*(X-x0)-np.sin(phi)*(Y-y0)
    Yp=np.sin(phi)*(X-x0)+np.cos(phi)*(Y-y0)

    x0n=z0*np.tan(theta)

    #Xp=np.cos(phi)*X-np.sin(phi)*Y-x0
    #Yp=np.sin(phi)*X+np.cos(phi)*Y-y0

    (z,r)=conv2(Xp,Yp,x0n,z0,theta)
    return (z,r)

def gauss_beam(r,z,w0,lam):
    zr=w0**2.0/lam*np.pi
    A=1.0+(0.0+1.0j)*z/zr
    return 1.0/A*np.exp(-r**2.0/(w0**2.0*A))*np.exp((0.0+2.0j)*np.pi/lam*z)

def gaussian(X,Y,x0,y0,z0,thetax,thetay,w0,lam):
    (Z,R)=FULL_CONV(X,Y,x0,y0,z0,thetax,thetay)
    G=gauss_beam(R,Z,w0,lam)
    return G




    
