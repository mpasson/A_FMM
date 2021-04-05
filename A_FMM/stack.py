import numpy as np
import A_FMM.sub_sm as sub
from A_FMM.scattering import S_matrix
import matplotlib.pyplot as plt
import copy
from A_FMM.layer import layer_empty_st
from matplotlib.backends.backend_pdf import PdfPages
try:
    from multiprocessing import Pool
except ModuleNotFoundError:
    print('WARNING: multiprocessing not available')
    

class stack:
    def __init__(self,layers=[],d=[]):
        if len(layers)!=len(d):
            raise ValueError('Different number of layers and thicknesses')
        self.N=len(layers)
        self.layers=layers
        self.d=d
        self.NPW=self.layers[0].D
        self.G=self.layers[0].G
        self.Nyx=self.layers[0].Nyx

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
            self.ex=ex
            FX=np.zeros((self.NPW,self.NPW),complex)
            for i in range(self.NPW):
                for j in range(self.NPW):
                    FX[i,j]=sub.fou_t(self.G[i][0]-self.G[j][0],ex)*(self.G[i][1]==self.G[j][1])
            for lay in self.lay_list:
                lay.add_transform_matrix(ex=ex,FX=FX)
        if (ey!=0.0):
            self.ey=ey
            FY=np.zeros((self.NPW,self.NPW),complex)
            for i in range(self.NPW):
                for j in range(self.NPW):
                    FY[i,j]=sub.fou_t(self.G[i][1]-self.G[j][1],ex)*(self.G[i][0]==self.G[j][0])
            for lay in self.lay_list:
                lay.add_transform_matrix(ey=ey,FY=FY)

    def transform_complex(self,ex=0.0,ey=0.0):
        g=1.0/(1-1j)
        if (ex!=0.0):
            self.ex=ex
            FX=np.zeros((self.NPW,self.NPW),complex)
            for i in range(self.NPW):
                for j in range(self.NPW):
                    FX[i,j]=sub.fou_complex_t(self.G[i][0]-self.G[j][0],ex,g)*(self.G[i][1]==self.G[j][1])
            for lay in self.lay_list:
                lay.add_transform_matrix(ex=ex,FX=FX)
        if (ey!=0.0):
            self.ey=ey
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

    def plot_stack(self,pdf=None,N=100,dz=0.01,y=0.0,func=np.abs, cmap='viridis'):
        X=np.linspace(-0.5,0.5,N)
        EPS=[]
        for (lay,d) in zip(self.layers,self.d):
            #EPSt=sum([sub.fou(lay.G[i][0],lay.G[i][1],lay.creator.x_list,lay.creator.y_list,lay.creator.eps_lists)*np.exp((0+2j)*np.pi*(lay.G[i][0]*X+lay.G[i][1]*y)) for i in range(lay.D)])
            EPSt=sum([lay.FOUP[i,lay.D//2]*np.exp((0+2j)*np.pi*(lay.G[i][0]*X+lay.G[i][1]*y)) for i in range(lay.D)])
            for i in range(int(d/dz)):
                EPS.append(EPSt)
        EPS=np.array(EPS)
        plt.figure()
        plt.imshow(func(EPS).T,origin='lower',extent=[0.0,sum(self.d),-0.5,0.5], cmap=plt.get_cmap(cmap))
        plt.colorbar()
        if isinstance(pdf,PdfPages):
            pdf.savefig()
        elif isinstance(pdf, str):
            pdf=pdf+'_y=%3.2f.pdf' % (y)
            a=PdfPages(pdf)
            a.savefig()
            a.close()
        if pdf is not None: plt.close()


    def plot_stack_y(self,nome='cross_section_Y',N=100,dz=0.01,x=0.0,func=np.abs):
        nome=nome+'_y=%3.2f.pdf' % (x)
        Y=np.linspace(-0.5,0.5,N)
        EPS=[]
        for (lay,d) in zip(self.layers,self.d):
            EPSt=sum([sub.fou(lay.G[i][0],lay.G[i][1],lay.creator.x_list,lay.creator.y_list,lay.creator.eps_lists)*np.exp((0+2j)*np.pi*(lay.G[i][0]*x+lay.G[i][1]*Y)) for i in range(lay.D)])
            for i in range(int(d/dz)):
                EPS.append(EPSt)
        EPS=np.array(EPS)
        out=PdfPages(nome)
        plt.figure()
        plt.imshow(func(EPS).T,origin='lower',extent=[0.0,sum(self.d),-0.5,0.5])
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

    def fourier(self,threads=1):
        p=Pool(threads)
        mat_list=p.map(layer_empty_st.fourier,self.lay_list)
        for lay,FOUP,INV,EPS1,EPS2 in zip(self.lay_list,mat_list):
            lay.FOUP=FOUP
            lay.INV=INV
            lay.EPS1=EPS1
            lay.EPS2=EPS2
        del mat_list

            

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

    def solve_serial(self,k0,kx=0.0,ky=0.0):
        lay1=self.layers[0]
        lay1.mode(k0,kx=kx,ky=ky)
        lay1.get_P_norm()
        self.S=S_matrix(2*self.NPW)
        for i in range(1,self.N):
            lay2=self.layers[i]
            lay2.mode(k0,kx=kx,ky=ky)
            self.S.add(lay1.interface(lay2))
            self.S.add_uniform(lay2,self.d[i])
            if i!=1 and i!=self.N:
                lay1.clear()
            lay1=lay2
        lay2.mode(k0,kx=kx,ky=ky)
        lay2.get_P_norm()


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

    def get_prop(self,u,list_lay,d=None):
        dic={}
        u1,d2=np.zeros((2*self.NPW),complex),np.zeros((2*self.NPW),complex)
        u1=u
        if d!=None:
            d2=d
        (u2,d1)=self.S.output(u1,d2)
        lay=self.layers[0]
        d=self.d[0]
        if 0 in list_lay:
            P=lay.get_Poynting(u1,d1)
            dic[0]=P
        #intermediate layers
        S1=copy.deepcopy(self.int_matrices[self.int_list.index(self.interfaces[0])])
        for i in range(1,self.N-1):
            S2=S_matrix(S1.N)
            for l in range(i,self.N-1):
                S2.add_uniform(self.layers[l],self.d[l])
                S2.add(self.int_matrices[self.int_list.index(self.interfaces[l])])
            if i in list_lay:
                (ul,dl)=S1.int_f_tot(S2,u1,d2)
                P=self.layers[i].get_Poynting(ul,dl)
                dic[i]=P
            S1.add_uniform(self.layers[i],self.d[i])
            S1.add(self.int_matrices[self.int_list.index(self.interfaces[i])])
        lay=self.layers[-1]
        d=self.d[-1]
        if self.N-1 in list_lay:
            P=lay.get_Poynting(u2,d2)
            dic[self.N-1]=P
        return dic

    def get_energybalance(self,u,d=None):
        u1,d2,e=np.zeros((2*self.NPW),complex),np.zeros((2*self.NPW),complex),np.zeros((2*self.NPW),complex)
        u1=u
        PN=self.layers[0].get_Poynting(u1,e)
        if d is not None:
            d2=d
            PN-=self.layers[-1].get_Poynting(e,d2)
        (u2,d1)=self.S.output(u1,d2)
        P1=self.layers[0].get_Poynting(u1,d1)
        P2=self.layers[-1].get_Poynting(u2,d2)
        return [P1/PN,P2/PN,(P1-P2)/PN]

    def get_inout(self,u,d=None):
        u1,d2=np.zeros((2*self.NPW),complex),np.zeros((2*self.NPW),complex)
        u1=u
        if d!=None:
            d2=d
        (u2,d1)=self.S.output(u1,d2)
        dic={}        
        P=self.layers[0].get_Poynting(u1,d1)
        dic['top']=(u1,d1,P)
        P=self.layers[-1].get_Poynting(u2,d2)
        dic['bottom']=(u2,d2,P)
        return dic

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
            raise RuntimeError('structure not solved yet')
        self.layers=[i for i in reversed(self.layers)]


    def bloch_modes(self):
        [self.BW,self.BV]=self.S.S_modes()
        self.Bk=-(0.0+1j)*np.log(self.BW)/self.tot_thick
        #reorder modes
        ind=np.argsort((0.0+1.0j)*self.Bk)
        self.BW=self.BW[ind]
        self.Bk=self.Bk[ind]
        self.BV[:,:]=self.BV[:,ind]        
        return self.Bk
        

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



    def plot_E(self,i=0,dz=0.01,pdf=None,N=100,y=0.0,func=np.real,s=1,ordered='yes',title=None, cmap='jet'):
        u1,d2=np.zeros((2*self.NPW),complex),np.zeros((2*self.NPW),complex)
        if ordered=='yes':
            u1[np.argsort(self.layers[0].W)[-i-1]]=1.0+0.0j
        else:
            u1[i]=1.0+0.0j
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
        if pdf is not None:
            if isinstance(pdf, PdfPages):
                out = pdf
            else:
                out=PdfPages(pdf)
        plt.figure()
        if title!=None:
            plt.suptitle(title)
        plt.subplot(211)
        plt.title('Ex')
        plt.imshow(func(Ex).T,origin='lower',extent=[0.0,sum(self.d),-0.5,0.5],cmap=plt.get_cmap(cmap))
        plt.colorbar()
        plt.subplot(212)
        plt.title('Ey')
        plt.imshow(func(Ey).T,origin='lower',extent=[0.0,sum(self.d),-0.5,0.5],cmap=plt.get_cmap(cmap))
        plt.colorbar()
        #plt.savefig('field.png',dpi=900)
        if pdf is not None:
            out.savefig()
            plt.close()
            if isinstance(pdf, str): out.close()        
        return None


    def writeE(self,i=1,filename='field.out',dz=0.01,N=100,y=0.0,func=np.real,s=1,ordered='yes'):
        f=open(filename,'w')
        u1,d2=np.zeros((2*self.NPW),complex),np.zeros((2*self.NPW),complex)
        if ordered=='yes':
            u1[np.argsort(self.layers[0].W)[-i]]=1.0+0.0j
        else:
            u1[i]=1.0+0.0j
        (u2,d1)=self.S.output(u1,d2)
        x=np.linspace(-s*0.5,s*0.5,s*N)
        ind=range(2*self.NPW)
        [X,I]=np.meshgrid(x,ind)
        Ex,Ey=[],[]
        #first layer
        lay=self.layers[0]
        d=self.d[0]
        Emx,Emy=np.zeros(np.shape(X),complex),np.zeros(np.shape(X),complex)
        zz=0.0
        for j in range(self.NPW):
            Emx=np.add(Emx,lay.V[j,I]*np.exp((0.0+2.0j)*np.pi*((lay.G[j][0]+lay.kx)*X+(lay.G[j][1]+lay.ky)*y)))
            Emy=np.add(Emy,lay.V[j+self.NPW,I]*np.exp((0.0+2.0j)*np.pi*((lay.G[j][0]+lay.kx)*X+(lay.G[j][1]+lay.ky)*y)))
        for z in np.arange(-d,0.0,dz):
            Em=np.add(u1*np.exp((0.0+2.0j)*np.pi*lay.k0*lay.gamma*z),d1*np.exp(-(0.0+2.0j)*np.pi*lay.k0*lay.gamma*z))
            Ex=np.dot(Em,Emx)
            Ey=np.dot(Em,Emy)
            for i in range(len(x)):
                f.write(8*'%12.6f' % (x[i],zz,Ex[i].real,Ex[i].imag,abs(Ex[i]),Ey[i].real,Ey[i].imag,abs(Ey[i]))+'\n')
            f.write('\n')
            zz+=dz
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
                Ex=np.dot(Em,Emx)
                Ey=np.dot(Em,Emy)
                for ii in range(len(x)):
                    f.write(8*'%12.6f' % (x[ii],zz,Ex[ii].real,Ex[ii].imag,abs(Ex[ii]),Ey[ii].real,Ey[ii].imag,abs(Ey[ii]))+'\n')
                f.write('\n')
                zz+=dz
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
            for i in range(len(x)):
                f.write(8*'%12.6f' % (x[i],zz,Ex[i].real,Ex[i].imag,abs(Ex[i]),Ey[i].real,Ey[i].imag,abs(Ey[i]))+'\n')
            f.write('\n')
            zz+=dz
        f.close()

    def plot_E_plane(self,i,jlay,z,N=100,pdf=None,pdfname=None,func=np.real,s=1,ordered='yes',title=None):
        u1,d2=np.zeros((2*self.NPW),complex),np.zeros((2*self.NPW),complex)
        if ordered=='yes':
            u1[np.argsort(self.layers[0].W)[-i]]=1.0+0.0j
        else:
            u1[i]=1.0+0.0j
        (u2,d1)=self.S.output(u1,d2)
        [X,Y]=np.meshgrid(np.linspace(-s*0.5,s*0.5,s*N),np.linspace(-s*0.5*self.layers[jlay].Nyx,s*0.5*self.layers[jlay].Nyx,s*N))
        S1=copy.deepcopy(self.int_matrices[self.int_list.index(self.interfaces[0])])
        S2=S_matrix(S1.N)
        for l in range(1,jlay):
            S1.add_uniform(self.layers[l],self.d[l])
            S1.add(self.int_matrices[self.int_list.index(self.interfaces[l])])
        for l in range(jlay,self.N-1):
            S2.add_uniform(self.layers[l],self.d[l])
            S2.add(self.int_matrices[self.int_list.index(self.interfaces[l])])
        (ul,dl)=S1.int_f(S2,u1)
        Emx_l,Emy_l=[],[]
        for i in range(2*self.NPW):
            Emx,Emy=np.zeros(np.shape(X),complex),np.zeros(np.shape(X),complex)
            for j in range(self.NPW):
                Emx=np.add(Emx,self.layers[jlay].V[j,i]*np.exp((0.0+2.0j)*np.pi*((self.layers[jlay].G[j][0]+self.layers[jlay].kx)*X+(self.layers[jlay].G[j][1]+self.layers[jlay].ky)*Y)))
                Emy=np.add(Emy,self.layers[jlay].V[j+self.NPW,i]*np.exp((0.0+2.0j)*np.pi*((self.layers[jlay].G[j][0]+self.layers[jlay].kx)*X+(self.layers[jlay].G[j][1]+self.layers[jlay].ky)*Y)))
            Emx_l.append(Emx)
            Emy_l.append(Emy)         
        Em=np.add(ul*np.exp((0.0+2.0j)*np.pi*self.layers[jlay].k0*self.layers[jlay].gamma*z*self.d[jlay]),dl*np.exp(-(0.0+2.0j)*np.pi*self.layers[jlay].k0*self.layers[jlay].gamma*z*self.d[jlay]))
        Ex,Ey=np.zeros(np.shape(X),complex),np.zeros(np.shape(X),complex)
        for i in range(2*self.NPW):
            Ex=np.add(Ex,Em[i]*Emx_l[i])
            Ey=np.add(Ey,Em[i]*Emy_l[i])
        if pdf==None:
            if pdfname!=None:
                out=PdfPages(pdfname+'.pdf')
            else:
                out=PdfPages('E_plane.pdf')
        else:
            out=pdf
        plt.figure()
        if title!=None:
            plt.suptitle(title)
        plt.subplot(211)
        plt.title('Ex')
        plt.imshow(func(Ex),origin='lower',extent=[-0.5,0.5,-0.5,0.5],cmap='jet')
        plt.colorbar()
        plt.subplot(212)
        plt.title('Ey')
        plt.imshow(func(Ey),origin='lower',extent=[-0.5,0.5,-0.5,0.5],cmap='jet')
        plt.colorbar()
        #plt.savefig('field.png',dpi=900)
        out.savefig(dpi=900)
        plt.close()
        if pdf==None:
            out.close()
        return None




    def plot_EY(self,i=1,dz=0.01,pdf=None,pdfname=None,N=100,x=0.0,func=np.real,s=1,ordered='yes',title=None):
        u1,d2=np.zeros((2*self.NPW),complex),np.zeros((2*self.NPW),complex)
        if ordered=='yes':
            u1[np.argsort(self.layers[0].W)[-i]]=1.0+0.0j
        else:
            u1[i]=1.0+0.0j
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
            out=PdfPages('EY.pdf')
        else:
            out=pdf
        plt.figure()
        if title!=None:
            plt.suptitle(title)
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


    def plot_E_periodic(self,ii,r=1,dz=0.01,pdf=None,N=100,y=0.0,func=np.real,s=1,title=None):
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
        if title!=None:
            plt.suptitle(title)
        #plt.savefig('field.png',dpi=900)
        out.savefig(dpi=900)
        plt.close()
        if pdf==None:
            out.close()
        return None

    def writeE_periodic_XZ(self,ii,r=1,filename='fieldE_XZ.out',dz=0.01,N=100,y=0.0,s=1.0):
        [u,d]=np.split(self.BV[:,ii],2)
        d=d*self.BW[ii]
        try:
            ex=self.ex
            if s==1.0:
                x=np.linspace(-0.5,0.5,N)
                xl=sub.t_dir(x,ex)
            else:
                xl=np.linspace(-s*0.5,s*0.5,N)
                x=sub.t_inv(xl,ex)
        except AttributeError:
                x=np.linspace(-0.5,0.5,N)
                xl=np.linspace(-0.5,0.5,N)
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
            start=0.0 if i==1 else dz
            for z in np.arange(start,self.d[i],dz):
                Em=np.add(ul*np.exp((0.0+2.0j)*np.pi*self.layers[i].k0*self.layers[i].gamma*z),dl*np.exp(-(0.0+2.0j)*np.pi*self.layers[i].k0*self.layers[i].gamma*z))
                Ex.append(np.dot(Em,Emx))
                Ey.append(np.dot(Em,Emy))
            S1.add_uniform(self.layers[i],self.d[i])
            S1.add(self.int_matrices[self.int_list.index(self.interfaces[i])])
        Ex,Ey=np.array(Ex),np.array(Ey)
        #print ii,np.abs([self.BW[ii]**k for k in range(r)])
        Ex=np.vstack([self.BW[ii]**k*Ex for k in range(r)])
        Ey=np.vstack([self.BW[ii]**k*Ey for k in range(r)])
        f=open(filename,'w')
        f.write('#    x           xt         z           ReEx        ImEx        AbsEx       ReEy        ImEy        AbsEy \n')
        for i in range(np.shape(Ex)[0]):
            for j in range(np.shape(Ex)[1]):
                f.write(9*'%15.6e' % (x[j],xl[j],dz*i,Ex[i,j].real,Ex[i,j].imag,abs(Ex[i,j]),Ey[i,j].real,Ey[i,j].imag,abs(Ey[i,j])) +'\n')
            f.write('\n')
        f.close()


    def writeE_periodic_YZ(self,ii,r=1,filename='fieldE_YZ.out',dz=0.01,N=100,x=0.0,s=1.0):
        [u,d]=np.split(self.BV[:,ii],2)
        d=d*self.BW[ii]
        try:
            ey=self.ey
            if s==1.0:
                y=np.linspace(-0.5,0.5,N)
                yl=sub.t_dir(y,ey)
            else:
                yl=np.linspace(-s*0.5,s*0.5,N)
                y=sub.t_inv(yl,ey)
        except AttributeError:
                y=np.linspace(-0.5,0.5,N)
                yl=np.linspace(-0.5,0.5,N)
        ind=range(2*self.NPW)
        [Y,I]=np.meshgrid(y,ind)
        Ex,Ey=[],[]
        #intermediate layers
        S1=copy.deepcopy(self.int_matrices[self.int_list.index(self.interfaces[0])])
        for i in range(1,self.N-1):
            S2=S_matrix(S1.N)
            for l in range(i,self.N-1):
                S2.add_uniform(self.layers[l],self.d[l])
                S2.add(self.int_matrices[self.int_list.index(self.interfaces[l])])
            (ul,dl)=S1.int_complete(S2,u,d)
            Emx,Emy=np.zeros(np.shape(Y),complex),np.zeros(np.shape(Y),complex)
            for j in range(self.NPW):
                Emx=np.add(Emx,self.layers[i].V[j,I]*np.exp((0.0+2.0j)*np.pi*((self.layers[i].G[j][0]+self.layers[i].kx)*x+(self.layers[i].G[j][1]+self.layers[i].ky)*Y)))
                Emy=np.add(Emy,self.layers[i].V[j+self.NPW,I]*np.exp((0.0+2.0j)*np.pi*((self.layers[i].G[j][0]+self.layers[i].kx)*x+(self.layers[i].G[j][1]+self.layers[i].ky)*Y)))
            start=0.0 if i==1 else dz
            for z in np.arange(start,self.d[i],dz):
                Em=np.add(ul*np.exp((0.0+2.0j)*np.pi*self.layers[i].k0*self.layers[i].gamma*z),dl*np.exp(-(0.0+2.0j)*np.pi*self.layers[i].k0*self.layers[i].gamma*z))
                Ex.append(np.dot(Em,Emx))
                Ey.append(np.dot(Em,Emy))
            S1.add_uniform(self.layers[i],self.d[i])
            S1.add(self.int_matrices[self.int_list.index(self.interfaces[i])])
        Ex,Ey=np.array(Ex),np.array(Ey)
        #print ii,np.abs([self.BW[ii]**k for k in range(r)])
        Ex=np.vstack([self.BW[ii]**k*Ex for k in range(r)])
        Ey=np.vstack([self.BW[ii]**k*Ey for k in range(r)])
        f=open(filename,'w')
        f.write('#    y           yt         z           ReEx        ImEx        AbsEx       ReEy        ImEy        AbsEy \n')
        for i in range(np.shape(Ex)[0]):
            for j in range(np.shape(Ex)[1]):
                f.write(9*'%15.6e' % (y[j]*self.Nyx,yl[j]*self.Nyx,dz*i,Ex[i,j].real,Ex[i,j].imag,abs(Ex[i,j]),Ey[i,j].real,Ey[i,j].imag,abs(Ey[i,j])) +'\n')
            f.write('\n')
        f.close()


    def writeE_periodic_XY(self,ii,jlay,z,filename='fieldE_XY.out',N=100,s=1.0):
        [u,d]=np.split(self.BV[:,ii],2)
        d=d*self.BW[ii]
        try:
            ex=self.ex
            if s==1.0:
                x=np.linspace(-0.5,0.5,N)
                xl=sub.t_dir(x,ex)
            else:
                xl=np.linspace(-s*0.5,s*0.5,N)
                x=sub.t_inv(xl,ex)
        except AttributeError:
                x=np.linspace(-0.5,0.5,N)
                xl=np.linspace(-0.5,0.5,N)
        try:
            ey=self.ey
            if s==1.0:
                y=np.linspace(-0.5,0.5,N)
                yl=sub.t_dir(y,ey)
            else:
                yl=np.linspace(-s*0.5,s*0.5,N)
                y=sub.t_inv(yl,ey)
        except AttributeError:
                y=np.linspace(-0.5,0.5,N)
                yl=np.linspace(-0.5,0.5,N)
        ind=range(2*self.NPW)
        [X,Y]=np.meshgrid(x,y)
        S1=copy.deepcopy(self.int_matrices[self.int_list.index(self.interfaces[0])])
        S2=S_matrix(S1.N)
        for l in range(1,jlay):
            S1.add_uniform(self.layers[l],self.d[l])
            S1.add(self.int_matrices[self.int_list.index(self.interfaces[l])])
        for l in range(jlay,self.N-1):
            S2.add_uniform(self.layers[l],self.d[l])
            S2.add(self.int_matrices[self.int_list.index(self.interfaces[l])])
        (ul,dl)=S1.int_complete(S2,u,d)
        Emx_l,Emy_l=[],[]
        for i in range(2*self.NPW):
            Emx,Emy=np.zeros(np.shape(X),complex),np.zeros(np.shape(X),complex)
            for j in range(self.NPW):
                Emx=np.add(Emx,self.layers[jlay].V[j,i]*np.exp((0.0+2.0j)*np.pi*((self.layers[jlay].G[j][0]+self.layers[jlay].kx)*X+(self.layers[jlay].G[j][1]+self.layers[jlay].ky)*Y)))
                Emy=np.add(Emy,self.layers[jlay].V[j+self.NPW,i]*np.exp((0.0+2.0j)*np.pi*((self.layers[jlay].G[j][0]+self.layers[jlay].kx)*X+(self.layers[jlay].G[j][1]+self.layers[jlay].ky)*Y)))
            Emx_l.append(Emx)
            Emy_l.append(Emy)         
        Em=np.add(ul*np.exp((0.0+2.0j)*np.pi*self.layers[jlay].k0*self.layers[jlay].gamma*z*self.d[jlay]),dl*np.exp(-(0.0+2.0j)*np.pi*self.layers[jlay].k0*self.layers[jlay].gamma*z*self.d[jlay]))
        Ex,Ey=np.zeros(np.shape(X),complex),np.zeros(np.shape(X),complex)
        for i in range(2*self.NPW):
            Ex=np.add(Ex,Em[i]*Emx_l[i])
            Ey=np.add(Ey,Em[i]*Emy_l[i])
        f=open(filename,'w')
        #f.write('#    y           yt         z           ReEx        ImEx        AbsEx       ReEy        ImEy        AbsEy \n')
        f.write('#    x           xt         y           yt          ReEx        ImEx        AbsEx       ReEy        ImEy        AbsEy \n')
        for i in range(np.shape(Ex)[0]):
            for j in range(np.shape(Ex)[1]):
                f.write(10*'%15.6e' % (x[i],xl[i],y[j]*self.Nyx,yl[j]*self.Nyx,Ex[i,j].real,Ex[i,j].imag,abs(Ex[i,j]),Ey[i,j].real,Ey[i,j].imag,abs(Ey[i,j])) +'\n')
            f.write('\n')
        f.close()





        
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
        print(st)
        print(22*'_')
        print('| INT argument')
        for i in att:
            if type(i[1]) is int:
                print('|%10s%10s' % (i[0],str(i[1])))
        print('| Float argument')
        for i in att:
            if type(i[1]) is float:
                print('|%10s%10s' % (i[0],str(i[1])))
        for i in att:
            if type(i[1]) is np.float64:
                print('|%10s%10s' % (i[0],str(i[1])))
        print('| BOOL argument')
        for i in att:
            if type(i[1]) is bool:
                print('|%10s%10s' % (i[0],str(i[1])))
        print('| Array argument')
        for i in att:
            if type(i[1]) is np.ndarray:
                print('|%10s%10s' % (i[0],str(np.shape(i[1]))))
        print('| List argument')
        for i in att:
            if type(i[1]) is list:
                print('|%12s%8s' % (i[0],str(len(i[1]))))
        print('')
        try:
            print('lay list:')
            for s in self.lay_list:
                print(s)

            print('layers:')
            for s in self.layers:
                print(s)

            print('int_list:')
            for s in self.int_list:
                print(s)

            print('interfaces:')
            for s in self.interfaces:
                print(s)
        except AttributeError:
            print('No list yet, call conut_interface before inspect')
