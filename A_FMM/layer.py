import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import sub_sm as sub
from matplotlib.backends.backend_pdf import PdfPages
from scattering import S_matrix
import copy


class layer:    
    def __init__(self,Nx,Ny,creator,Nyx=1.0):
        self.Nx=Nx
        self.Ny=Ny
        self.G=sub.createG(self.Nx,self.Ny)
        self.D=len(self.G)
        self.creator=copy.deepcopy(creator)
        self.Nyx=Nyx

        #self.FOUP=linalg.toeplitz([sub.fou(self.G[i][0]-self.G[0][0],self.G[i][1]-self.G[0][1],self.creator.x_list,self.creator.y_list,self.creator.eps_lists) for i in range(self.D)])
        self.FOUP=sub.create_epsilon(self.G,self.creator.x_list,self.creator.y_list,self.creator.eps_lists)
        self.INV=linalg.inv(self.FOUP)
#        self.EPS_INV=linalg.inv(linalg.toeplitz([sub.fou_v(self.G[i][0]-self.G[0][0],self.G[i][1]-self.G[0][1],self.creator.x_list,self.creator.y_list,self.creator.eps_lists) for i in range(self.D)]))
#        self.REC=linalg.toeplitz([sub.fou_v(self.G[i][0]-self.G[0][0],self.G[i][1]-self.G[0][1],self.creator.x_list,self.creator.y_list,self.creator.eps_lists) for i in range(self.D)])

        self.EPS1=sub.fou_xy(self.Nx,self.Ny,self.G,self.creator.x_list,self.creator.y_list,self.creator.eps_lists)
        self.EPS2=sub.fou_yx(self.Nx,self.Ny,self.G,self.creator.x_list,self.creator.y_list,self.creator.eps_lists)

        self.TX=False
        self.TY=False

    def inspect(self,st=''):
        att=sub.get_user_attributes(self)
        print st
        print 22*'_'
        print '| INT argument'
        for i in att:
            if type(i[1]) is int:
                print '|%10s%10s' % (i[0],str(i[1]))
        print '| Float argument'
        for i in att:
            if type(i[1]) is float:
                print '|%10s%10s' % (i[0],str(i[1]))
        for i in att:
            if type(i[1]) is np.float64:
                print '|%10s%10s' % (i[0],str(i[1]))
        print '| BOOL argument'
        for i in att:
            if type(i[1]) is bool:
                print '|%10s%10s' % (i[0],str(i[1]))
        print '| Array argument'
        for i in att:
            if type(i[1]) is np.ndarray:
                print '|%10s%10s' % (i[0],str(np.shape(i[1])))
        print ''

    def eps_plot(self,pdf=None,N=200,s=1.0):
        [X,Y]=np.meshgrid(np.linspace(-s*0.5,s*0.5,s*N),np.linspace(-s*0.5,s*0.5,int(s*N*self.Nyx)))
        EPS=np.zeros((N,N),complex)
#        for i in range(self.D):
#            EPS+=sub.fou(self.G[i][0],self.G[i][1],self.creator.x_list,self.creator.y_list,self.creator.eps_lists)*np.exp(-(0+2j)*np.pi*(self.G[i][0]*X+self.G[i][1]*Y))
        EPS=sum([sub.fou(self.G[i][0],self.G[i][1],self.creator.x_list,self.creator.y_list,self.creator.eps_lists)*np.exp((0+2j)*np.pi*(self.G[i][0]*X+self.G[i][1]*Y)) for i in range(self.D)])
        plt.figure()
        #plt.imshow(np.real(EPS),aspect='auto',extent=[-s*0.5,s*0.5,-self.Nyx*s*0.5,self.Nyx*s*0.5])
        plt.imshow(np.real(EPS),extent=[-s*0.5,s*0.5,-self.Nyx*s*0.5,self.Nyx*s*0.5])
        plt.colorbar()
        if (pdf==None):
            plt.show()
        else:
            a=PdfPages(pdf+'.pdf')
            a.savefig()
            a.close()
        plt.close()

    def trasform(self,ex=0,ey=0):
        if (ex!=0):
            self.TX=True
            self.ex=ex
#            self.FX=linalg.toeplitz([sub.fou_t(self.G[i][0]-self.G[0][0],ex) for i in range(self.D)])
            self.FX=np.zeros((self.D,self.D),complex)
            for i in range(self.D):
                for j in range(self.D):
                    self.FX[i,j]=sub.fou_t(self.G[i][0]-self.G[j][0],ex)*(self.G[i][1]==self.G[j][1])
        if (ey!=0):
            self.TY=True
            self.ey=ey
#            self.FY=linalg.toeplitz([sub.fou_t(self.G[i][0]-self.G[0][0],ex) for i in range(self.D)])
            self.FY=np.zeros((self.D,self.D),complex)
            for i in range(self.D):
                for j in range(self.D):
                    self.FY[i,j]=sub.fou_t(self.G[i][1]-self.G[j][1],ey)*(self.G[i][0]==self.G[j][0])


    def trasform_complex(self,ex=0,ey=0):
        g=1.0/(1-1j)
        if (ex!=0):
            self.TX=True
            self.ex=ex
#            self.FX=linalg.toeplitz([sub.fou_t(self.G[i][0]-self.G[0][0],ex) for i in range(self.D)])
            self.FX=np.zeros((self.D,self.D),complex)
            for i in range(self.D):
                for j in range(self.D):
                    self.FX[i,j]=sub.fou_complex_t(self.G[i][0]-self.G[j][0],ex,g)*(self.G[i][1]==self.G[j][1])
        if (ey!=0):
            self.TY=True
            self.ey=ey
#            self.FY=linalg.toeplitz([sub.fou_t(self.G[i][0]-self.G[0][0],ex) for i in range(self.D)])
            self.FY=np.zeros((self.D,self.D),complex)
            for i in range(self.D):
                for j in range(self.D):
                    self.FY[i,j]=sub.fou_complex_t(self.G[i][1]-self.G[j][1],ey,g)*(self.G[i][0]==self.G[j][0])

    def add_transform_matrix(self,ex=0.0,FX=None,ey=0.0,FY=None):
        if (ex!=0):
            self.TX=True
            self.ex=ex
            self.FX=FX
        if (ey!=0):
            self.TY=True
            self.ey=ey
            self.FY=FY
        
        


    def mode(self,k0,kx=0.0,ky=0.0,v=1):
        self.k0=k0
        self.kx=kx
        self.ky=ky
        (k1,k2)=sub.createK(self.G,k0,kx=kx,ky=ky,Nyx=self.Nyx)
        if self.TX:
            #print np.shape(self.FX),np.shape(self.k1)
            k1=np.dot(self.FX,k1)
        if self.TY:
            k2=np.dot(self.FY,k2)
        self.GH,self.M=sub.create_2order_new(self.D,k1,k2,self.INV,self.EPS1,self.EPS2)
        if (v!=0):
            [self.W,self.V]=linalg.eig(self.M)
            self.gamma=np.sqrt(self.W)*np.sign(np.angle(self.W)+0.5*np.pi)
            if np.any(np.real(self.gamma)+np.imag(self.gamma)<=0.0):
                print 'Warining: wrong complex root'
#            self.gamma=np.sqrt(self.W)
#            self.gamma=self.gamma*np.sign(np.imag(self.gamma))
#            if np.any(np.imag(self.gamma)<=0.0):
#                print 'Warining: negative imaginary part'
            if np.any(np.abs(self.gamma)<=0.0):
                print 'Warining: gamma=0'
            
            self.VH=np.dot(self.GH,self.V/self.gamma)
        else:
            self.W=linalg.eigvals(self.M)


    def clear(self):
        self.W=None
        self.V=None
        self.gamma=None

                
    def mat_plot(self,name,N=100,s=1):
        save=PdfPages(name+'.pdf')

        [X,Y]=np.meshgrid(np.linspace(-s*0.5,s*0.5,s*N),np.linspace(-s*0.5,s*0.5,s*N))
        EPS=np.zeros((N,N),complex)
        EPS=sum([sub.fou(self.G[i][0],self.G[i][1],self.creator.x_list,self.creator.y_list,self.creator.eps_lists)*np.exp((0+2j)*np.pi*(self.G[i][0]*X+self.G[i][1]*Y)) for i in range(self.D)])

        plt.figure()
        plt.title('epsilon real')
        plt.imshow(np.real(EPS),aspect=1,origin='lower')
        plt.colorbar()
        save.savefig()
        plt.close()


        plt.figure()
        plt.title('epsilon')
        plt.imshow(np.abs(self.FOUP),aspect='auto',interpolation='nearest')
        plt.colorbar()
        save.savefig()
        plt.close()

        #plt.figure()
        #plt.title('epsilon_inv')
        #plt.imshow(np.abs(self.EPS_INV),aspect='auto',interpolation='nearest')
        #plt.colorbar()
        #save.savefig()
        #plt.close()

        plt.figure()
        plt.title('Epsilon 1')
        plt.imshow(np.abs(self.EPS1),aspect='auto',interpolation='nearest')
        plt.colorbar()
        save.savefig()
        plt.close()

        plt.figure()
        plt.title('Epsilon 2')
        plt.imshow(np.abs(self.EPS2),aspect='auto',interpolation='nearest')
        plt.colorbar()
        save.savefig()
        plt.close()


        plt.figure()
        plt.title('diff eps-eps1')
        plt.imshow(np.abs(self.FOUP-self.EPS1),aspect='auto',interpolation='nearest')
        plt.colorbar()
        save.savefig()
        plt.close()

        plt.figure()
        plt.title('INV')
        plt.imshow(np.abs(np.abs(self.INV)),aspect='auto',interpolation='nearest')
        plt.colorbar()
        save.savefig()
        plt.close()

        if self.TX:
            plt.figure()
            plt.title('FX')
            plt.imshow(np.abs(np.abs(self.FX)),aspect='auto',interpolation='nearest')
            plt.colorbar()
            save.savefig()
            plt.close()

        if self.TY:
            plt.figure()
            plt.title('FY')
            plt.imshow(np.abs(np.abs(self.FY)),aspect='auto',interpolation='nearest')
            plt.colorbar()
            save.savefig()
            plt.close()

        save.close()

    def plot_Ham(self,pdf):
        N=np.shape(self.M)[0]
        plt.figure()
        plt.title('k0:%5.3f kx:%5.3f ky:%5.3f' % (self.k0,self.kx,self.ky))
        plt.imshow(np.abs(np.abs(self.M)),aspect='auto',interpolation='nearest')
        plt.colorbar()
        pdf.savefig()
        plt.close()

    def plot_E(self,pdf,i,N=100,s=1,func=np.abs):
        j=np.argsort(self.W)[-i]
        [X,Y]=np.meshgrid(np.linspace(-s*0.5,s*0.5,s*N),np.linspace(-s*0.5,s*0.5,s*N))
        [WEy,WEx]=np.split(self.V[:,j],2)
        Ey,Ex=np.zeros((N,N),dtype='complex'),np.zeros((N,N),dtype='complex')
        for i in range(self.D):
            EXP=np.exp((0+2j)*np.pi*((self.G[i][0]+self.kx)*X+(self.G[i][1]+self.ky)*Y))
            Ey+=WEy[i]*EXP
            Ex+=WEx[i]*EXP
        plt.figure()
        plt.title('k0:%5.3f kx:%5.3f ky:%5.3f' % (self.k0,self.kx,self.ky))
        plt.subplot(211)
        plt.imshow(func(Ex),aspect=1,extent=[-s*0.5,s*0.5,-s*0.5,s*0.5])
        plt.title('Ex')
        plt.colorbar()
        plt.subplot(212)
        plt.imshow(func(Ey),aspect=1,extent=[-s*0.5,s*0.5,-s*0.5,s*0.5])
        plt.title('Ey')
        plt.colorbar()
        pdf.savefig()
        plt.close()

    def plot_H(self,pdf,i,N=100,s=1,func=np.abs):
        j=np.argsort(self.W)[-i]
        [X,Y]=np.meshgrid(np.linspace(-s*0.5,s*0.5,s*N),np.linspace(-s*0.5,s*0.5,s*N))
        [WEy,WEx]=np.split(self.VH[:,j],2)
        Ey,Ex=np.zeros((N,N),dtype='complex'),np.zeros((N,N),dtype='complex')
        for i in range(self.D):
            EXP=np.exp((0+2j)*np.pi*((self.G[i][0]+self.kx)*X+(self.G[i][1]+self.ky)*Y))
            Ey+=WEy[i]*EXP
            Ex+=WEx[i]*EXP
        plt.figure()
        plt.title('k0:%5.3f kx:%5.3f ky:%5.3f' % (self.k0,self.kx,self.ky))
        plt.subplot(211)
        plt.imshow(func(Ex),aspect=1,extent=[-s*0.5,s*0.5,-s*0.5,s*0.5])
        plt.title('Hx')
        plt.colorbar()
        plt.subplot(212)
        plt.imshow(func(Ey),aspect=1,extent=[-s*0.5,s*0.5,-s*0.5,s*0.5])
        plt.title('Hy')
        plt.colorbar()
        pdf.savefig()
        plt.close()


    def plot_field(self,pdf,i,N=100,s=1,func=np.abs):
        j=np.argsort(self.W)[-i]
        [X,Y]=np.meshgrid(np.linspace(-s*0.5,s*0.5,s*N),np.linspace(-s*0.5,s*0.5,s*N))
        [WEx,WEy]=np.split(self.V[:,j],2)
        [WHx,WHy]=np.split(self.VH[:,j],2)
        Ex,Ey=np.zeros((N,N),dtype='complex'),np.zeros((N,N),dtype='complex')
        Hx,Hy=np.zeros((N,N),dtype='complex'),np.zeros((N,N),dtype='complex')
        for i in range(self.D):
            EXP=np.exp((0+2j)*np.pi*((self.G[i][0]+self.kx)*X+(self.G[i][1]+self.ky)*Y))
#            Ex+=WEx[i]*EXP
#            Ey+=WEy[i]*EXP
#            Hx+=WHx[i]*EXP
#            Hy+=WHy[i]*EXP
            Ex=np.add(Ex,np.dot(WEx[i],EXP))
            Ey=np.add(Ey,np.dot(WEy[i],EXP))
            Hx=np.add(Hx,np.dot(WHx[i],EXP))
            Hy=np.add(Hy,np.dot(WHy[i],EXP))
        plt.subplot(221)
        plt.imshow(func(Ex),aspect=1,extent=[-s*0.5,s*0.5,-s*0.5,s*0.5],origin='lower')
        plt.title('Ex')
        plt.colorbar()
        plt.subplot(222)
        plt.imshow(func(Ey),aspect=1,extent=[-s*0.5,s*0.5,-s*0.5,s*0.5],origin='lower')
        plt.title('Ey')
        plt.colorbar()
        plt.subplot(223)
        plt.imshow(func(Hx),aspect=1,extent=[-s*0.5,s*0.5,-s*0.5,s*0.5],origin='lower')
        plt.title('Hx')
        plt.colorbar()
        plt.subplot(224)
        plt.imshow(func(Hy),aspect=1,extent=[-s*0.5,s*0.5,-s*0.5,s*0.5],origin='lower')
        plt.title('Hy')
        plt.colorbar()
        pdf.savefig()
        plt.close()


    def get_field(self,x,y,i,func=np.abs):
        j=np.argsort(self.W)[-i]
        [WEx,WEy]=np.split(self.V[:,j],2)
        [WHx,WHy]=np.split(self.VH[:,j],2)
        Ex,Ey=0.0,0.0
        Hx,Hy=0.0,0.0
        for i in range(self.D):
            EXP=np.exp((0+2j)*np.pi*((self.G[i][0]+self.kx)*x+(self.G[i][1]+self.ky)*y))
            Ex+=WEx[i]*EXP
            Ey+=WEy[i]*EXP
            Hx+=WHx[i]*EXP
            Hy+=WHy[i]*EXP
        return func(np.array([Ex,Ey,Hx,Hy]))


    def plot_Et(self,pdf,i,N=100,sx=1,sy=1,func=np.abs):
        j=np.argsort(self.W)[-i]
        x,y=np.linspace(-sx*0.5,sx*0.5,sx*N),np.linspace(-sy*0.5,sy*0.5,sy*N)
        if self.TX:
            x=sub.t_inv(x,self.ex)
        if self.TY:
            y=sub.t_inv(y,self.ey)
        [X,Y]=np.meshgrid(x,y)
        [WEy,WEx]=np.split(self.V[:,j],2)
        Ey,Ex=np.zeros((sy*N,sx*N),dtype='complex'),np.zeros((sy*N,sx*N),dtype='complex')
        for i in range(self.D):
            EXP=np.exp((0+2j)*np.pi*((self.G[i][0]+self.kx)*X+(self.G[i][1]+self.ky)*Y))
            Ey+=WEy[i]*EXP
            Ex+=WEx[i]*EXP
        plt.figure()
        plt.title('k0:%5.3f kx:%5.3f ky:%5.3f' % (self.k0,self.kx,self.ky))
        plt.subplot(211)
        plt.imshow(func(Ex),extent=[-sx*0.5,sx*0.5,-sy*0.5,sy*0.5])
        plt.title('Ex')
        plt.colorbar()
        plt.subplot(212)
        plt.imshow(func(Ey),extent=[-sx*0.5,sx*0.5,-sy*0.5,sy*0.5])
        plt.title('Ey')
        plt.colorbar()
        pdf.savefig()
        plt.close()

    

    def get_P_norm(self):
        [VEx,VEy]=np.split(self.V,2)
        [VHx,VHy]=np.split(self.VH,2)
        self.P_norm=np.sum(VEx*np.conj(VHy)-VEy*np.conj(VHx),0).real
    
 
    def T_interface(self,lay):
        T1=np.dot(linalg.inv(lay.V),self.V)
        T2=np.dot(linalg.inv(lay.VH),self.VH)
        T11= 0.5*(T1 + T2)
        T12= 0.5*(T1 - T2)
        T21= 0.5*(T1 - T2)
        T22= 0.5*(T1 + T2)
        T=np.vstack([np.hstack([T11,T12]),np.hstack([T21,T22])])
        return T

    def T_prop(self,d):
        I1=np.diag(np.exp((0+1j)*self.k0*self.gamma*d))
        I2=np.diag(np.exp(-(0+1j)*self.k0*self.gamma*d))
        I=np.zeros((2*self.D,2*self.D),complex)
        T=np.vstack([np.hstack([I1,I]),np.hstack([I,I2])])
        return T
        
#newer version, should be faster
    def interface(self,lay):
        S=S_matrix(2*self.D)
        T1=np.dot(linalg.inv(lay.V),self.V)
        T2=np.dot(linalg.inv(lay.VH),self.VH)
        T11= 0.5*(T1 + T2)
        T12= 0.5*(T1 - T2)
        #T21= 0.5*(T1 - T2)
        #T22= 0.5*(T1 + T2)
        #T=np.vstack([np.hstack([T11,T12]),np.hstack([T21,T22])])
        Tm=linalg.inv(T11)
        S.S11=T11-np.dot(np.dot(T12,Tm),T12)
        S.S12=np.dot(T12,Tm)
        S.S21=-np.dot(Tm,T12)
        S.S22=Tm
        return S



class layer_ani_diag(layer):
    def __init__(self,Nx,Ny,creator_x,creator_y,creator_z,Nyx=1.0):
        self.Nx=Nx
        self.Ny=Ny
        self.G=sub.createG(self.Nx,self.Ny)
        self.D=len(self.G)
        self.creator=[copy.deepcopy(creator_x),copy.deepcopy(creator_y),copy.deepcopy(creator_z)]
        self.Nyx=Nyx
        
        self.FOUP=sub.create_epsilon(self.G,self.creator[2].x_list,self.creator[2].y_list,self.creator[2].eps_lists)
        self.INV=linalg.inv(self.FOUP)

        self.EPS1=sub.fou_xy(self.Nx,self.Ny,self.G,self.creator[0].x_list,self.creator[0].y_list,self.creator[0].eps_lists)
        self.EPS2=sub.fou_yx(self.Nx,self.Ny,self.G,self.creator[1].x_list,self.creator[1].y_list,self.creator[1].eps_lists)

        self.TX=False
        self.TY=False


class layer_num(layer):
    def __init__(self,Nx,Ny,func,args=(),Nyx=1.0,NX=1024,NY=1024):
        self.Nx=Nx
        self.Ny=Ny
        self.G=sub.createG(self.Nx,self.Ny)
        self.D=len(self.G)
        self.Nyx=Nyx
        self.func=func
        self.args=args
        #print args
        
        self.FOUP=sub.num_fou(func,args,self.G,NX=NX,NY=NY,Nyx=self.Nyx)
        self.INV=linalg.inv(self.FOUP)

        #Still to be defined
        self.EPS1=sub.num_fou_xy(self.func,self.args,self.Nx,self.Ny,self.G,NX=1024,NY=1024,Nyx=self.Nyx)
        self.EPS2=sub.num_fou_yx(self.func,self.args,self.Nx,self.Ny,self.G,NX=1024,NY=1024,Nyx=self.Nyx)
        

        self.TX=False
        self.TY=False


    def eps_plot(self,pdf=None,N=200,s=1.0):
        [X,Y]=np.meshgrid(np.linspace(-s*0.5,s*0.5,s*N),np.linspace(-s*0.5,s*0.5,int(s*N*self.Nyx)))
        [XX,YY]=np.meshgrid(np.linspace(-0.5,0.5,1024),np.linspace(-0.5,0.5,1024))
        #F=self.func(XX,YY)
        F=self.func(XX % 1.0-0.5,(YY % 1.0-0.5)/self.Nyx,*self.args)
        FOU=np.fft.fft2(F)/1024.0/1024.0
        EPS=np.zeros((N,N),complex)
        EPS=sum([FOU[self.G[i][0],self.G[i][1]]*np.exp((0+2j)*np.pi*(self.G[i][0]*X+self.G[i][1]*Y)) for i in range(self.D)])
        plt.figure()
        plt.imshow(np.real(EPS),extent=[-s*0.5,s*0.5,-self.Nyx*s*0.5,self.Nyx*s*0.5])
        plt.colorbar()
        if (pdf==None):
            plt.show()
        else:
            a=PdfPages(pdf+'.pdf')
            a.savefig()
            a.close()
        plt.close()


    def mat_plot(self,name,N=100,s=1):
        save=PdfPages(name+'.pdf')

        [X,Y]=np.meshgrid(np.linspace(-s*0.5,s*0.5,s*N),np.linspace(-s*0.5,s*0.5,s*N))
        [XX,YY]=np.meshgrid(np.linspace(-0.5,0.5,1024),np.linspace(-0.5,0.5,1024))
        #F=self.func(XX,YY)
        F=self.func(XX % 1.0-0.5,(YY % 1.0-0.5)/self.Nyx,*self.args)
        FOU=np.fft.fft2(F)/1024.0/1024.0
        EPS=np.zeros((N,N),complex)
        EPS=sum([FOU[self.G[i][0],self.G[i][1]]*np.exp((0+2j)*np.pi*(self.G[i][0]*X+self.G[i][1]*Y)) for i in range(self.D)])

        plt.figure()
        plt.title('epsilon real')
        plt.imshow(np.real(EPS),aspect=1,origin='lower')
        plt.colorbar()
        save.savefig()
        plt.close()


        plt.figure()
        plt.title('epsilon')
        plt.imshow(np.abs(self.FOUP),aspect='auto',interpolation='nearest')
        plt.colorbar()
        save.savefig()
        plt.close()

        #plt.figure()
        #plt.title('epsilon_inv')
        #plt.imshow(np.abs(self.EPS_INV),aspect='auto',interpolation='nearest')
        #plt.colorbar()
        #save.savefig()
        #plt.close()

        plt.figure()
        plt.title('Epsilon 1')
        plt.imshow(np.abs(self.EPS1),aspect='auto',interpolation='nearest')
        plt.colorbar()
        save.savefig()
        plt.close()

        plt.figure()
        plt.title('Epsilon 2')
        plt.imshow(np.abs(self.EPS2),aspect='auto',interpolation='nearest')
        plt.colorbar()
        save.savefig()
        plt.close()


        plt.figure()
        plt.title('diff eps-eps1')
        plt.imshow(np.abs(self.FOUP-self.EPS1),aspect='auto',interpolation='nearest')
        plt.colorbar()
        save.savefig()
        plt.close()

        plt.figure()
        plt.title('INV')
        plt.imshow(np.abs(np.abs(self.INV)),aspect='auto',interpolation='nearest')
        plt.colorbar()
        save.savefig()
        plt.close()
        save.close()













