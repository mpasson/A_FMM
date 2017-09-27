import numpy as np
import sub_sm as sub
from scattering import S_matrix
import matplotlib.pyplot as plt
import copy
from matplotlib.backends.backend_pdf import PdfPages


class stack:
    def __init__(self,layers=[],d=[]):
        if len(layers)!=len(d):
            raise ValueError('Different number of layers and thicknesses')
        self.N=len(layers)
        self.layers=layers
        self.d=d
        self.NPW=self.layers[0].D
        self.G=self.layers[0].G

    def add_layer(lay,d):
        self.layers.append(lay)
        self.d.append(d)
        self.N+=1

    def add_transform(self,ex=0.0,ey=0.0):
        self.ex=ex
        self.ey=ey
        for lay in self.layers:
            lay.trasform(ex,ey)

    def add_transform_complex(self,ex=0.0,ey=0.0):
        self.ex=ex
        self.ey=ey
        for lay in self.layers:
            lay.trasform_complex(ex,ey)

    def transform(self,ex=0.0,ey=0.0):
        if (ex!=0.0):
            FX=np.zeros((self.NPW,self.NPW),complex)
            for i in range(self.NPW):
                for j in range(self.NPW):
                    FX[i,j]=sub.fou_t(self.G[i][0]-self.G[j][0],ex)*(self.G[i][1]==self.G[j][1])
            for lay in self.lay_list:
                lay.add_transform_matrix(ex=ex,FX=FX)
        if (ey!=0.0):
            FY=np.zeros((self.NPW,self.NPW),complex)
            for i in range(self.NPW):
                for j in range(self.NPW):
                    FY[i,j]=sub.fou_t(self.G[i][1]-self.G[j][1],ex)*(self.G[i][0]==self.G[j][0])
            for lay in self.lay_list:
                lay.add_transform_matrix(ey=ey,FY=FY)

    def transform_complex(self,ex=0.0,ey=0.0):
        g=1.0/(1-1j)
        if (ex!=0.0):
            FX=np.zeros((self.NPW,self.NPW),complex)
            for i in range(self.NPW):
                for j in range(self.NPW):
                    FX[i,j]=sub.fou_complex_t(self.G[i][0]-self.G[j][0],ex,g)*(self.G[i][1]==self.G[j][1])
            for lay in self.lay_list:
                lay.add_transform_matrix(ex=ex,FX=FX)
        if (ey!=0.0):
            FY=np.zeros((self.NPW,self.NPW),complex)
            for i in range(self.NPW):
                for j in range(self.NPW):
                    FY[i,j]=sub.fou_complex_t(self.G[i][1]-self.G[j][1],ex,g)*(self.G[i][0]==self.G[j][0])
            for lay in self.lay_list:
                lay.add_transform_matrix(ey=ey,FY=FY)


    def mat_plot(self,N=100,s=1):
        n=1
        for lay in self.layers:
            lay.mat_plot('layer_%i' % (n),N=N,s=s)
            n+=1

    def plot_stack(self,nome='cross_section_X',N=100,dx=0.01,y=0.0):
        nome=nome+'_y=%3.2f.pdf' % (y)
        X=np.linspace(-0.5,0.5,N)
        EPS=[]
        for (lay,d) in zip(self.layers,self.d):
            EPSt=sum([sub.fou(lay.G[i][0],lay.G[i][1],lay.creator.x_list,lay.creator.y_list,lay.creator.eps_lists)*np.exp((0+2j)*np.pi*(lay.G[i][0]*X+lay.G[i][1]*y)) for i in range(lay.D)])
            for i in range(int(d/dx)):
                EPS.append(EPSt)
        EPS=np.array(EPS)
        out=PdfPages(nome)
        plt.figure()
        plt.imshow(np.abs(EPS).T,origin='lower',extent=[0.0,sum(self.d),-0.5,0.5])
        plt.colorbar()
        out.savefig(dpi=900)
        plt.close()
        out.close()


    def plot_stack_y(self,nome='cross_section_Y',N=100,dx=0.01,x=0.0):
        nome=nome+'_y=%3.2f.pdf' % (x)
        Y=np.linspace(-0.5,0.5,N)
        EPS=[]
        for (lay,d) in zip(self.layers,self.d):
            EPSt=sum([sub.fou(lay.G[i][0],lay.G[i][1],lay.creator.x_list,lay.creator.y_list,lay.creator.eps_lists)*np.exp((0+2j)*np.pi*(lay.G[i][0]*x+lay.G[i][1]*Y)) for i in range(lay.D)])
            for i in range(int(d/dx)):
                EPS.append(EPSt)
        EPS=np.array(EPS)
        out=PdfPages(nome)
        plt.figure()
        plt.imshow(np.abs(EPS).T,origin='lower',extent=[0.0,sum(self.d),-0.5,0.5])
        plt.colorbar()
        out.savefig()
        plt.close()
        out.close()



    def count_interface(self):
        self.tot_thick=sum(self.d)
        self.lay_list=[]
        for lay in self.layers:
            if not lay in self.lay_list:
                self.lay_list.append(lay)
        self.int_list=[]
        self.interfaces=[]
        for i in range(self.N-1):
            T_inter=(self.layers[i],self.layers[i+1])
            if not T_inter in self.int_list:
                self.int_list.append(T_inter)
            self.interfaces.append(T_inter)

            

    def solve(self,k0,kx=0.0,ky=0.0):
        for lay in self.lay_list:
            lay.mode(k0,kx=kx,ky=ky)
            #lay.get_P_norm()
        self.layers[0].get_P_norm()
        self.layers[-1].get_P_norm()
        self.int_matrices=[]
        for i in self.int_list:
            self.int_matrices.append(i[0].interface(i[1]))
        self.S=copy.deepcopy(self.int_matrices[0])
        for i in range(1,self.N-1):        
            self.S.add_uniform(self.layers[i],self.d[i])
            self.S.add(self.int_matrices[self.int_list.index(self.interfaces[i])])


    def solve_lay(self,k0,kx=0.0,ky=0.0):
        for lay in self.lay_list:
            lay.mode(k0,kx=kx,ky=ky)
            #lay.get_P_norm()
        self.layers[0].get_P_norm()
        self.layers[-1].get_P_norm()

    def solve_S(self,k0,kx=0.0,ky=0.0):
        self.int_matrices=[]
        for i in self.int_list:
            self.int_matrices.append(i[0].interface(i[1]))
        self.S=copy.deepcopy(self.int_matrices[0])
        for i in range(1,self.N-1):        
            self.S.add_uniform(self.layers[i],self.d[i])
            self.S.add(self.int_matrices[self.int_list.index(self.interfaces[i])])

    def get_R(self,i,j,ordered='yes'):
        return self.S.get_R(i,j,self.layers[0],ordered=ordered)

    def get_T(self,i,j,ordered='yes'):
        return self.S.get_T(i,self.layers[0],j,self.layers[-1],ordered=ordered)

    def get_PR(self,i,j,ordered='yes'):
        return self.S.get_PR(i,j,self.layers[0],ordered=ordered)

    def get_PT(self,i,j,ordered='yes'):
        return self.S.get_PT(i,self.layers[0],j,self.layers[-1],ordered=ordered)

    def get_el(self,sel,i,j):
        io=np.argsort(self.layers[0].W)[-i]
        jo=np.argsort(self.layers[-1].W)[-j]
        if sel=='11':
            return self.S.S11[io,jo]
        elif sel=='12':
            return self.S.S12[io,jo]
        elif sel=='21':
            return self.S.S21[io,jo]
        elif sel=='22':
            return self.S.S22[io,jo]

    def mode_T(self,uin,uout):
        d1=np.zeros((2*self.NPW),complex)
        [u1,d1]=self.S.output(uin,d1)
        return np.abs(np.dot(np.conj(u1),uout))**2



    def double(self):
        try:
            self.S.add(self.S)
        except AttributeError:
            raise RuntimeError('structure not solved yet')


    def join(self,st2):
        try:
            self.S
        except AttributeError:
            raise RuntimeError('structure 1 not solved yet')
        try:
            st2.S
        except AttributeError:
            raise RuntimeError('structure 2 not solved yet')
        self.S.add(st2.S)
        l1=self.layers[:-1]
        l2=st2.layers[1:]
        self.layers=l1+l2
        

    def flip(self):
        try:
            S=copy.deepcopy(self.S)
            self.S.S11=S.S22
            self.S.S22=S.S11
            self.S.S12=S.S21
            self.S.S21=S.S12
        except AttributeError:
            raise RuntimeError('structure 2 not solved yet')
        self.layers=[i for i in reversed(self.layers)]


    def bloch_modes(self):
        [self.BW,self.BV]=self.S.S_modes()
        self.Bk=-(0.0+1j)*np.log(self.BW)/self.tot_thick
        #reorder modes
        ind=np.argsort((0.0+1.0j)*self.Bk)
        self.BW=self.BW[ind]
        self.Bk=self.Bk[ind]
        self.BV[:,:]=self.BV[:,ind]        
        

    def plot_Ey(self,i,dz=0.01,pdf=None,N=100,y=0.0,func=np.real,s=1):
        u1,d2=np.zeros((2*self.NPW),complex),np.zeros((2*self.NPW),complex)
        u1[np.argsort(self.layers[0].W)[-i]]=1.0+0.0j
        (u2,d1)=self.S.output(u1,d2)
        x=np.linspace(-s*0.5,s*0.5,s*N)
        ind=range(2*self.NPW)
        [X,I]=np.meshgrid(x,ind)
        Em=np.zeros(np.shape(X),complex)
        E=[]
        #first layer
        lay=self.layers[0]
        d=self.d[0]
        Em=np.zeros(np.shape(X),complex)
        for j in range(self.NPW):
            Em=np.add(Em,lay.V[j+self.NPW,I]*np.exp((0.0+2.0j)*np.pi*((lay.G[j][0]+lay.kx)*X+(lay.G[j][1]+lay.ky)*y)))
        for z in np.arange(-d,0.0,dz):
            Emx=np.add(u1*np.exp((0.0+2.0j)*np.pi*lay.k0*lay.gamma*z),d1*np.exp(-(0.0+2.0j)*np.pi*lay.k0*lay.gamma*z))
            Ez=np.dot(Emx,Em)
            E.append(Ez)
        #intermediate layers
        S1=copy.deepcopy(self.int_matrices[self.int_list.index(self.interfaces[0])])
        for i in range(1,self.N-1):
            S2=S_matrix(S1.N)
            for l in range(i,self.N-1):
                S2.add_uniform(self.layers[l],self.d[l])
                S2.add(self.int_matrices[self.int_list.index(self.interfaces[l])])
            (ul,dl)=S1.int_f(S2,u1)
            Em=np.zeros(np.shape(X),complex)
            for j in range(self.NPW):
                Em=np.add(Em,self.layers[i].V[j+self.NPW,I]*np.exp((0.0+2.0j)*np.pi*((self.layers[i].G[j][0]+self.layers[i].kx)*X+(self.layers[i].G[j][1]+self.layers[i].ky)*y)))
            for z in np.arange(0.0,self.d[i],dz):
                Emx=np.add(ul*np.exp((0.0+2.0j)*np.pi*self.layers[i].k0*self.layers[i].gamma*z),dl*np.exp(-(0.0+2.0j)*np.pi*self.layers[i].k0*self.layers[i].gamma*z-self.d[i]))
                Ez=np.dot(Emx,Em)
                E.append(Ez)
            S1.add_uniform(self.layers[i],self.d[i])
            S1.add(self.int_matrices[self.int_list.index(self.interfaces[i])])
        #last layer
        lay=self.layers[-1]
        d=self.d[-1]
        Em=np.zeros(np.shape(X),complex)
        for j in range(self.NPW):
            Em=np.add(Em,lay.V[j+self.NPW,I]*np.exp((0.0+2.0j)*np.pi*((lay.G[j][0]+lay.kx)*X+(lay.G[j][1]+lay.ky)*y)))
        for z in np.arange(0.0,d,dz):
            Emx=np.add(u2*np.exp((0.0+2.0j)*np.pi*lay.k0*lay.gamma*z),d2*np.exp(-(0.0+2.0j)*np.pi*lay.k0*lay.gamma*z))
            Ez=np.dot(Emx,Em)
            E.append(Ez)

        if pdf==None:
            out=PdfPages('Ey.pdf')
        else:
            out=pdf
        plt.figure()
        plt.imshow(func(E).T,origin='lower',extent=[0.0,sum(self.d),-0.5,0.5],cmap='jet')
        #plt.colorbar()
        #plt.savefig('field.png',dpi=900)
        out.savefig(dpi=900)
        plt.close()
        if pdf==None:
            out.close()
        return None

    def plot_Ex(self,i,dz=0.01,pdf=None,N=100,y=0.0,func=np.real,s=1):
        u1,d2=np.zeros((2*self.NPW),complex),np.zeros((2*self.NPW),complex)
        u1[np.argsort(self.layers[0].W)[-i]]=1.0+0.0j
        (u2,d1)=self.S.output(u1,d2)
        x=np.linspace(-s*0.5,s*0.5,s*N)
        ind=range(2*self.NPW)
        [X,I]=np.meshgrid(x,ind)
        Em=np.zeros(np.shape(X),complex)
        E=[]
        #first layer
        lay=self.layers[0]
        d=self.d[0]
        Em=np.zeros(np.shape(X),complex)
        for j in range(self.NPW):
            Em=np.add(Em,lay.V[j,I]*np.exp((0.0+2.0j)*np.pi*((lay.G[j][0]+lay.kx)*X+(lay.G[j][1]+lay.ky)*y)))
        for z in np.arange(-d,0.0,dz):
            Emx=np.add(u1*np.exp((0.0+2.0j)*np.pi*lay.k0*lay.gamma*z),d1*np.exp(-(0.0+2.0j)*np.pi*lay.k0*lay.gamma*z))
            Ez=np.dot(Emx,Em)
            E.append(Ez)
        #intermediate layers
        S1=copy.deepcopy(self.int_matrices[self.int_list.index(self.interfaces[0])])
        for i in range(1,self.N-1):
            S2=S_matrix(S1.N)
            for l in range(i,self.N-1):
                S2.add_uniform(self.layers[l],self.d[l])
                S2.add(self.int_matrices[self.int_list.index(self.interfaces[l])])
            (ul,dl)=S1.int_f(S2,u1)
            Em=np.zeros(np.shape(X),complex)
            for j in range(self.NPW):
                Em=np.add(Em,self.layers[i].V[j,I]*np.exp((0.0+2.0j)*np.pi*((self.layers[i].G[j][0]+self.layers[i].kx)*X+(self.layers[i].G[j][1]+self.layers[i].ky)*y)))
            for z in np.arange(0.0,self.d[i],dz):
                Emx=np.add(ul*np.exp((0.0+2.0j)*np.pi*self.layers[i].k0*self.layers[i].gamma*z),dl*np.exp(-(0.0+2.0j)*np.pi*self.layers[i].k0*self.layers[i].gamma*z-self.d[i]))
                Ez=np.dot(Emx,Em)
                E.append(Ez)
            S1.add_uniform(self.layers[i],self.d[i])
            S1.add(self.int_matrices[self.int_list.index(self.interfaces[i])])
        #last layer
        lay=self.layers[-1]
        d=self.d[-1]
        Em=np.zeros(np.shape(X),complex)
        for j in range(self.NPW):
            Em=np.add(Em,lay.V[j,I]*np.exp((0.0+2.0j)*np.pi*((lay.G[j][0]+lay.kx)*X+(lay.G[j][1]+lay.ky)*y)))
        for z in np.arange(0.0,d,dz):
            Emx=np.add(u2*np.exp((0.0+2.0j)*np.pi*lay.k0*lay.gamma*z),d2*np.exp(-(0.0+2.0j)*np.pi*lay.k0*lay.gamma*z))
            Ez=np.dot(Emx,Em)
            E.append(Ez)
        
        if pdf==None:
            out=PdfPages('Ex.pdf')
        else:
            out=pdf
        plt.figure()
        plt.imshow(func(E).T,origin='lower',extent=[0.0,sum(self.d),-0.5,0.5],cmap='jet')
        #plt.colorbar()
        #plt.savefig('field.png',dpi=900)
        out.savefig(dpi=900)
        plt.close()
        if pdf==None:
            out.close()
        return None



    def plot_E(self,i=1,dz=0.01,pdf=None,pdfname=None,N=100,y=0.0,func=np.real,s=1):
        u1,d2=np.zeros((2*self.NPW),complex),np.zeros((2*self.NPW),complex)
        u1[np.argsort(self.layers[0].W)[-i]]=1.0+0.0j
        (u2,d1)=self.S.output(u1,d2)
        x=np.linspace(-s*0.5,s*0.5,s*N)
        ind=range(2*self.NPW)
        [X,I]=np.meshgrid(x,ind)
        Ex,Ey=[],[]
        #first layer
        lay=self.layers[0]
        d=self.d[0]
        Emx,Emy=np.zeros(np.shape(X),complex),np.zeros(np.shape(X),complex)
        for j in range(self.NPW):
            Emx=np.add(Emx,lay.V[j,I]*np.exp((0.0+2.0j)*np.pi*((lay.G[j][0]+lay.kx)*X+(lay.G[j][1]+lay.ky)*y)))
            Emy=np.add(Emy,lay.V[j+self.NPW,I]*np.exp((0.0+2.0j)*np.pi*((lay.G[j][0]+lay.kx)*X+(lay.G[j][1]+lay.ky)*y)))
        for z in np.arange(-d,0.0,dz):
            Em=np.add(u1*np.exp((0.0+2.0j)*np.pi*lay.k0*lay.gamma*z),d1*np.exp(-(0.0+2.0j)*np.pi*lay.k0*lay.gamma*z))
            Ex.append(np.dot(Em,Emx))
            Ey.append(np.dot(Em,Emy))
        #intermediate layers
        S1=copy.deepcopy(self.int_matrices[self.int_list.index(self.interfaces[0])])
        for i in range(1,self.N-1):
            S2=S_matrix(S1.N)
            for l in range(i,self.N-1):
                S2.add_uniform(self.layers[l],self.d[l])
                S2.add(self.int_matrices[self.int_list.index(self.interfaces[l])])
            (ul,dl)=S1.int_f(S2,u1)
            Emx,Emy=np.zeros(np.shape(X),complex),np.zeros(np.shape(X),complex)
            for j in range(self.NPW):
                Emx=np.add(Emx,self.layers[i].V[j,I]*np.exp((0.0+2.0j)*np.pi*((self.layers[i].G[j][0]+self.layers[i].kx)*X+(self.layers[i].G[j][1]+self.layers[i].ky)*y)))
                Emy=np.add(Emy,self.layers[i].V[j+self.NPW,I]*np.exp((0.0+2.0j)*np.pi*((self.layers[i].G[j][0]+self.layers[i].kx)*X+(self.layers[i].G[j][1]+self.layers[i].ky)*y)))
            for z in np.arange(0.0,self.d[i],dz):
                Em=np.add(ul*np.exp((0.0+2.0j)*np.pi*self.layers[i].k0*self.layers[i].gamma*z),dl*np.exp(-(0.0+2.0j)*np.pi*self.layers[i].k0*self.layers[i].gamma*z))
                Ex.append(np.dot(Em,Emx))
                Ey.append(np.dot(Em,Emy))
            S1.add_uniform(self.layers[i],self.d[i])
            S1.add(self.int_matrices[self.int_list.index(self.interfaces[i])])
        #last layer
        lay=self.layers[-1]
        d=self.d[-1]
        Emx,Emy=np.zeros(np.shape(X),complex),np.zeros(np.shape(X),complex)
        for j in range(self.NPW):
            Emx=np.add(Emx,lay.V[j,I]*np.exp((0.0+2.0j)*np.pi*((lay.G[j][0]+lay.kx)*X+(lay.G[j][1]+lay.ky)*y)))
            Emy=np.add(Emy,lay.V[j+self.NPW,I]*np.exp((0.0+2.0j)*np.pi*((lay.G[j][0]+lay.kx)*X+(lay.G[j][1]+lay.ky)*y)))
        for z in np.arange(0.0,d,dz):
            Em=np.add(u2*np.exp((0.0+2.0j)*np.pi*lay.k0*lay.gamma*z),d2*np.exp(-(0.0+2.0j)*np.pi*lay.k0*lay.gamma*z))
            Ex.append(np.dot(Em,Emx))
            Ey.append(np.dot(Em,Emy))
        if pdf==None:
            if pdfname!=None:
                out=PdfPages(pdfname+'.pdf')
            else:
                out=PdfPages('E.pdf')
        else:
            out=pdf
        plt.figure()
        plt.subplot(211)
        plt.title('Ex')
        plt.imshow(func(Ex).T,origin='lower',extent=[0.0,sum(self.d),-0.5,0.5],cmap='jet')
        plt.colorbar()
        plt.subplot(212)
        plt.title('Ey')
        plt.imshow(func(Ey).T,origin='lower',extent=[0.0,sum(self.d),-0.5,0.5],cmap='jet')
        plt.colorbar()
        #plt.savefig('field.png',dpi=900)
        out.savefig(dpi=900)
        plt.close()
        if pdf==None:
            out.close()
        return None


    def plot_EY(self,i=1,dz=0.01,pdf=None,N=100,x=0.0,func=np.real,s=1):
        u1,d2=np.zeros((2*self.NPW),complex),np.zeros((2*self.NPW),complex)
        u1[np.argsort(self.layers[0].W)[-i]]=1.0+0.0j
        (u2,d1)=self.S.output(u1,d2)
        y=np.linspace(-s*0.5,s*0.5,s*N)
        ind=range(2*self.NPW)
        [Y,I]=np.meshgrid(y,ind)
        Ex,Ey=[],[]
        #first layer
        lay=self.layers[0]
        d=self.d[0]
        Emx,Emy=np.zeros(np.shape(Y),complex),np.zeros(np.shape(Y),complex)
        for j in range(self.NPW):
            Emx=np.add(Emx,lay.V[j,I]*np.exp((0.0+2.0j)*np.pi*((lay.G[j][0]+lay.kx)*x+(lay.G[j][1]+lay.ky)*Y)))
            Emy=np.add(Emy,lay.V[j+self.NPW,I]*np.exp((0.0+2.0j)*np.pi*((lay.G[j][0]+lay.kx)*x+(lay.G[j][1]+lay.ky)*Y)))
        for z in np.arange(-d,0.0,dz):
            Em=np.add(u1*np.exp((0.0+2.0j)*np.pi*lay.k0*lay.gamma*z),d1*np.exp(-(0.0+2.0j)*np.pi*lay.k0*lay.gamma*z))
            Ex.append(np.dot(Em,Emx))
            Ey.append(np.dot(Em,Emy))
        #intermediate layers
        S1=copy.deepcopy(self.int_matrices[self.int_list.index(self.interfaces[0])])
        for i in range(1,self.N-1):
            S2=S_matrix(S1.N)
            for l in range(i,self.N-1):
                S2.add_uniform(self.layers[l],self.d[l])
                S2.add(self.int_matrices[self.int_list.index(self.interfaces[l])])
            (ul,dl)=S1.int_f(S2,u1)
            Emx,Emy=np.zeros(np.shape(Y),complex),np.zeros(np.shape(Y),complex)
            for j in range(self.NPW):
                Emx=np.add(Emx,self.layers[i].V[j,I]*np.exp((0.0+2.0j)*np.pi*((self.layers[i].G[j][0]+self.layers[i].kx)*x+(self.layers[i].G[j][1]+self.layers[i].ky)*Y)))
                Emy=np.add(Emy,self.layers[i].V[j+self.NPW,I]*np.exp((0.0+2.0j)*np.pi*((self.layers[i].G[j][0]+self.layers[i].kx)*x+(self.layers[i].G[j][1]+self.layers[i].ky)*Y)))
            for z in np.arange(0.0,self.d[i],dz):
                Em=np.add(ul*np.exp((0.0+2.0j)*np.pi*self.layers[i].k0*self.layers[i].gamma*z),dl*np.exp(-(0.0+2.0j)*np.pi*self.layers[i].k0*self.layers[i].gamma*z))
                Ex.append(np.dot(Em,Emx))
                Ey.append(np.dot(Em,Emy))
            S1.add_uniform(self.layers[i],self.d[i])
            S1.add(self.int_matrices[self.int_list.index(self.interfaces[i])])
        #last layer
        lay=self.layers[-1]
        d=self.d[-1]
        Emx,Emy=np.zeros(np.shape(Y),complex),np.zeros(np.shape(Y),complex)
        for j in range(self.NPW):
            Emx=np.add(Emx,lay.V[j,I]*np.exp((0.0+2.0j)*np.pi*((lay.G[j][0]+lay.kx)*x+(lay.G[j][1]+lay.ky)*Y)))
            Emy=np.add(Emy,lay.V[j+self.NPW,I]*np.exp((0.0+2.0j)*np.pi*((lay.G[j][0]+lay.kx)*x+(lay.G[j][1]+lay.ky)*Y)))
        for z in np.arange(0.0,d,dz):
            Em=np.add(u2*np.exp((0.0+2.0j)*np.pi*lay.k0*lay.gamma*z),d2*np.exp(-(0.0+2.0j)*np.pi*lay.k0*lay.gamma*z))
            Ex.append(np.dot(Em,Emx))
            Ey.append(np.dot(Em,Emy))
        if pdf==None:
            out=PdfPages('E.pdf')
        else:
            out=pdf
        plt.figure()
        plt.subplot(211)
        plt.title('Ex')
        plt.imshow(func(Ex).T,origin='lower',extent=[0.0,sum(self.d),-0.5,0.5],cmap='jet')
        #plt.colorbar()
        plt.subplot(212)
        plt.title('Ey')
        plt.imshow(func(Ey).T,origin='lower',extent=[0.0,sum(self.d),-0.5,0.5],cmap='jet')
        #plt.savefig('field.png',dpi=900)
        out.savefig(dpi=900)
        plt.close()
        if pdf==None:
            out.close()
        return None


    def plot_E_periodic(self,ii,r=1,dz=0.01,pdf=None,N=100,y=0.0,func=np.real,s=1):
        [u,d]=np.split(self.BV[:,ii],2)
        d=d*self.BW[ii]
        x=np.linspace(-s*0.5,s*0.5,s*N)
        ind=range(2*self.NPW)
        [X,I]=np.meshgrid(x,ind)
        Ex,Ey=[],[]
        #intermediate layers
        S1=copy.deepcopy(self.int_matrices[self.int_list.index(self.interfaces[0])])
        for i in range(1,self.N-1):
            S2=S_matrix(S1.N)
            for l in range(i,self.N-1):
                S2.add_uniform(self.layers[l],self.d[l])
                S2.add(self.int_matrices[self.int_list.index(self.interfaces[l])])
            (ul,dl)=S1.int_complete(S2,u,d)
            Emx,Emy=np.zeros(np.shape(X),complex),np.zeros(np.shape(X),complex)
            for j in range(self.NPW):
                Emx=np.add(Emx,self.layers[i].V[j,I]*np.exp((0.0+2.0j)*np.pi*((self.layers[i].G[j][0]+self.layers[i].kx)*X+(self.layers[i].G[j][1]+self.layers[i].ky)*y)))
                Emy=np.add(Emy,self.layers[i].V[j+self.NPW,I]*np.exp((0.0+2.0j)*np.pi*((self.layers[i].G[j][0]+self.layers[i].kx)*X+(self.layers[i].G[j][1]+self.layers[i].ky)*y)))
            for z in np.arange(0.0,self.d[i],dz):
                Em=np.add(ul*np.exp((0.0+2.0j)*np.pi*self.layers[i].k0*self.layers[i].gamma*z),dl*np.exp(-(0.0+2.0j)*np.pi*self.layers[i].k0*self.layers[i].gamma*z))
                Ex.append(np.dot(Em,Emx))
                Ey.append(np.dot(Em,Emy))
            S1.add_uniform(self.layers[i],self.d[i])
            S1.add(self.int_matrices[self.int_list.index(self.interfaces[i])])
        Ex,Ey=np.array(Ex),np.array(Ey)
        #print ii,np.abs([self.BW[ii]**k for k in range(r)])
        Ex=np.vstack([self.BW[ii]**k*Ex for k in range(r)])
        Ey=np.vstack([self.BW[ii]**k*Ey for k in range(r)])
        if pdf==None:
            out=PdfPages('E.pdf')
        else:
            out=pdf
        plt.figure()
        plt.subplot(211)
        plt.title('Ex')
        plt.imshow(func(Ex).T,origin='lower',extent=[0.0,r*sum(self.d),-0.5,0.5],cmap='jet')
        plt.colorbar()
        plt.subplot(212)
        plt.title('Ey')
        plt.imshow(func(Ey).T,origin='lower',extent=[0.0,r*sum(self.d),-0.5,0.5],cmap='jet')
        plt.colorbar()
        #plt.savefig('field.png',dpi=900)
        out.savefig(dpi=900)
        plt.close()
        if pdf==None:
            out.close()
        return None

    def create_input(self,dic):
        u=np.zeros((2*self.NPW),complex)
        for i in dic:
            u[np.argsort(self.layers[0].W)[-i]]=dic[i]
        return u


    def plot_E_general(self,u,d=None,dz=0.01,pdf=None,N=100,y=0.0,func=np.real,s=1):
        u1,d2=np.zeros((2*self.NPW),complex),np.zeros((2*self.NPW),complex)
        #u1[np.argsort(self.layers[0].W)[-i]]=1.0+0.0j
        u1=u
        if d!=None:
            d2=d
        (u2,d1)=self.S.output(u1,d2)
        x=np.linspace(-s*0.5,s*0.5,s*N)
        ind=range(2*self.NPW)
        [X,I]=np.meshgrid(x,ind)
        Ex,Ey=[],[]
        #first layer
        lay=self.layers[0]
        d=self.d[0]
        Emx,Emy=np.zeros(np.shape(X),complex),np.zeros(np.shape(X),complex)
        for j in range(self.NPW):
            Emx=np.add(Emx,lay.V[j,I]*np.exp((0.0+2.0j)*np.pi*((lay.G[j][0]+lay.kx)*X+(lay.G[j][1]+lay.ky)*y)))
            Emy=np.add(Emy,lay.V[j+self.NPW,I]*np.exp((0.0+2.0j)*np.pi*((lay.G[j][0]+lay.kx)*X+(lay.G[j][1]+lay.ky)*y)))
        for z in np.arange(-d,0.0,dz):
            Em=np.add(u1*np.exp((0.0+2.0j)*np.pi*lay.k0*lay.gamma*z),d1*np.exp(-(0.0+2.0j)*np.pi*lay.k0*lay.gamma*z))
            Ex.append(np.dot(Em,Emx))
            Ey.append(np.dot(Em,Emy))
        #intermediate layers
        S1=copy.deepcopy(self.int_matrices[self.int_list.index(self.interfaces[0])])
        for i in range(1,self.N-1):
            S2=S_matrix(S1.N)
            for l in range(i,self.N-1):
                S2.add_uniform(self.layers[l],self.d[l])
                S2.add(self.int_matrices[self.int_list.index(self.interfaces[l])])
            (ul,dl)=S1.int_f_tot(S2,u1,d2)
            Emx,Emy=np.zeros(np.shape(X),complex),np.zeros(np.shape(X),complex)
            for j in range(self.NPW):
                Emx=np.add(Emx,self.layers[i].V[j,I]*np.exp((0.0+2.0j)*np.pi*((self.layers[i].G[j][0]+self.layers[i].kx)*X+(self.layers[i].G[j][1]+self.layers[i].ky)*y)))
                Emy=np.add(Emy,self.layers[i].V[j+self.NPW,I]*np.exp((0.0+2.0j)*np.pi*((self.layers[i].G[j][0]+self.layers[i].kx)*X+(self.layers[i].G[j][1]+self.layers[i].ky)*y)))
            for z in np.arange(0.0,self.d[i],dz):
                Em=np.add(ul*np.exp((0.0+2.0j)*np.pi*self.layers[i].k0*self.layers[i].gamma*z),dl*np.exp(-(0.0+2.0j)*np.pi*self.layers[i].k0*self.layers[i].gamma*z))
                Ex.append(np.dot(Em,Emx))
                Ey.append(np.dot(Em,Emy))
            S1.add_uniform(self.layers[i],self.d[i])
            S1.add(self.int_matrices[self.int_list.index(self.interfaces[i])])
        #last layer
        lay=self.layers[-1]
        d=self.d[-1]
        Emx,Emy=np.zeros(np.shape(X),complex),np.zeros(np.shape(X),complex)
        for j in range(self.NPW):
            Emx=np.add(Emx,lay.V[j,I]*np.exp((0.0+2.0j)*np.pi*((lay.G[j][0]+lay.kx)*X+(lay.G[j][1]+lay.ky)*y)))
            Emy=np.add(Emy,lay.V[j+self.NPW,I]*np.exp((0.0+2.0j)*np.pi*((lay.G[j][0]+lay.kx)*X+(lay.G[j][1]+lay.ky)*y)))
        for z in np.arange(0.0,d,dz):
            Em=np.add(u2*np.exp((0.0+2.0j)*np.pi*lay.k0*lay.gamma*z),d2*np.exp(-(0.0+2.0j)*np.pi*lay.k0*lay.gamma*z))
            Ex.append(np.dot(Em,Emx))
            Ey.append(np.dot(Em,Emy))
        if pdf==None:
            out=PdfPages('E.pdf')
        else:
            out=pdf
        plt.figure()
        plt.subplot(211)
        plt.title('Ex')
        plt.imshow(func(Ex).T,origin='lower',extent=[0.0,sum(self.d),-0.5,0.5],cmap='jet')
        plt.colorbar()
        plt.subplot(212)
        plt.title('Ey')
        plt.imshow(func(Ey).T,origin='lower',extent=[0.0,sum(self.d),-0.5,0.5],cmap='jet')
        plt.colorbar()
        #plt.savefig('field.png',dpi=900)
        out.savefig(dpi=900)
        plt.close()
        if pdf==None:
            out.close()
        return None



    def line_E(self,i,x=0.0,y=0.0,dz=0.01):
        u1,d2=np.zeros((2*self.NPW),complex),np.zeros((2*self.NPW),complex)
        u1[np.argsort(self.layers[0].W)[-i]]=1.0+0.0j
        (u2,d1)=self.S.output(u1,d2)
        ind=range(2*self.NPW)
        I=ind
        X=x
        Ex,Ey,zl=[],[],[]
        D=np.cumsum(self.d)
        #first layer
        lay=self.layers[0]
        d=self.d[0]
        Emx,Emy=np.zeros(np.shape(X),complex),np.zeros(np.shape(X),complex)
        for j in range(self.NPW):
            Emx=np.add(Emx,lay.V[j,I]*np.exp((0.0+2.0j)*np.pi*((lay.G[j][0]+lay.kx)*X+(lay.G[j][1]+lay.ky)*y)))
            Emy=np.add(Emy,lay.V[j+self.NPW,I]*np.exp((0.0+2.0j)*np.pi*((lay.G[j][0]+lay.kx)*X+(lay.G[j][1]+lay.ky)*y)))
        for z in np.arange(-d,0.0,dz):
            Em=np.add(u1*np.exp((0.0+2.0j)*np.pi*lay.k0*lay.gamma*z),d1*np.exp(-(0.0+2.0j)*np.pi*lay.k0*lay.gamma*z))
            Ex.append(np.dot(Em,Emx))
            Ey.append(np.dot(Em,Emy))
            zl.append(D[0]+z)
        #intermediate layers
        S1=copy.deepcopy(self.int_matrices[self.int_list.index(self.interfaces[0])])
        for i in range(1,self.N-1):
            S2=S_matrix(S1.N)
            for l in range(i,self.N-1):
                S2.add_uniform(self.layers[l],self.d[l])
                S2.add(self.int_matrices[self.int_list.index(self.interfaces[l])])
            (ul,dl)=S1.int_f(S2,u1)
            Emx,Emy=np.zeros(np.shape(X),complex),np.zeros(np.shape(X),complex)
            for j in range(self.NPW):
                Emx=np.add(Emx,self.layers[i].V[j,I]*np.exp((0.0+2.0j)*np.pi*((self.layers[i].G[j][0]+self.layers[i].kx)*X+(self.layers[i].G[j][1]+self.layers[i].ky)*y)))
                Emy=np.add(Emy,self.layers[i].V[j+self.NPW,I]*np.exp((0.0+2.0j)*np.pi*((self.layers[i].G[j][0]+self.layers[i].kx)*X+(self.layers[i].G[j][1]+self.layers[i].ky)*y)))
            for z in np.arange(0.0,self.d[i],dz):
                Em=np.add(ul*np.exp((0.0+2.0j)*np.pi*self.layers[i].k0*self.layers[i].gamma*z),dl*np.exp(-(0.0+2.0j)*np.pi*self.layers[i].k0*self.layers[i].gamma*z))
                Ex.append(np.dot(Em,Emx))
                Ey.append(np.dot(Em,Emy))
                zl.append(D[i-1]+z)
            S1.add_uniform(self.layers[i],self.d[i])
            S1.add(self.int_matrices[self.int_list.index(self.interfaces[i])])
        #last layer
        lay=self.layers[-1]
        d=self.d[-1]
        Emx,Emy=np.zeros(np.shape(X),complex),np.zeros(np.shape(X),complex)
        for j in range(self.NPW):
            Emx=np.add(Emx,lay.V[j,I]*np.exp((0.0+2.0j)*np.pi*((lay.G[j][0]+lay.kx)*X+(lay.G[j][1]+lay.ky)*y)))
            Emy=np.add(Emy,lay.V[j+self.NPW,I]*np.exp((0.0+2.0j)*np.pi*((lay.G[j][0]+lay.kx)*X+(lay.G[j][1]+lay.ky)*y)))
        for z in np.arange(0.0,d,dz):
            Em=np.add(u2*np.exp((0.0+2.0j)*np.pi*lay.k0*lay.gamma*z),d2*np.exp(-(0.0+2.0j)*np.pi*lay.k0*lay.gamma*z))
            Ex.append(np.dot(Em,Emx))
            Ey.append(np.dot(Em,Emy))
            zl.append(D[-2]+z)

        return [np.array(zl),np.array(Ex),np.array(Ey)]



    def inspect(self,st='',details='no'):
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
        print '| List argument'
        for i in att:
            if type(i[1]) is list:
                print '|%12s%8s' % (i[0],str(len(i[1])))
        print ''
        try:
            print 'lay list:'
            for s in self.lay_list:
                print s

            print 'layers:'
            for s in self.layers:
                print s

            print 'int_list:'
            for s in self.int_list:
                print s

            print 'interfaces:'
            for s in self.interfaces:
                print s
        except AttributeError:
            print 'No list yet, call conut_interface before inspect'
