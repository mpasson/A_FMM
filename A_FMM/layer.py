import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import A_FMM.sub_sm as sub
from A_FMM.creator import Creator
from matplotlib.backends.backend_pdf import PdfPages
from A_FMM.scattering import S_matrix
import copy


class Layer:    
    """ Class for the definition of a single layer
    """
    def __init__(self,Nx,Ny,creator,Nyx=1.0):
        """Creator
        
        Args:
            Nx (int): truncation order in x direction
            Ny (int): truncation order in y direction
            Nyx (float): ratio between the cell's dimension in y and x (ay/ax)
        """
        self.Nx=Nx
        self.Ny=Ny
        self.NPW=(2*Nx+1)*(2*Ny+1)
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
        """Function for inspectig the attributes of a layer object
            
        Args:
            st (string): string to print before the inspection for identification
        """
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
        print('')

    def eps_plot(self,pdf=None,N=200,s=1):
        """Function for plotting the dielectric consstat rebuit from plane wave expansion

        Args:
            pdf (string or PdfPages): file for printing the the epsilon
                if a PdfPages object, the page is appended to the pdf
                if string, a pdf with that name is created
            N (int): number of points
            s (float): number of cell replicas to display (default 1)
        """
        [X,Y]=np.meshgrid(np.linspace(-s*0.5,s*0.5,s*N),np.linspace(-s*0.5,s*0.5,int(s*N*self.Nyx)))
        EPS=np.zeros((N,N),complex)
#        for i in range(self.D):
#            EPS+=sub.fou(self.G[i][0],self.G[i][1],self.creator.x_list,self.creator.y_list,self.creator.eps_lists)*np.exp(-(0+2j)*np.pi*(self.G[i][0]*X+self.G[i][1]*Y))
        EPS=sum([sub.fou(self.G[i][0],self.G[i][1],self.creator.x_list,self.creator.y_list,self.creator.eps_lists)*np.exp((0+2j)*np.pi*(self.G[i][0]*X+self.G[i][1]*Y)) for i in range(self.D)])
        plt.figure()
        #plt.imshow(np.real(EPS),aspect='auto',extent=[-s*0.5,s*0.5,-self.Nyx*s*0.5,self.Nyx*s*0.5])
        plt.imshow(np.real(EPS),extent=[-s*0.5,s*0.5,-self.Nyx*s*0.5,self.Nyx*s*0.5],origin='lower')
        plt.colorbar()
        if (pdf==None):
            plt.show()
        elif isinstance(pdf,PdfPages):
            pdf.savefig()
        else:
            a=PdfPages(pdf+'.pdf')
            a.savefig()
            a.close()
        plt.close()

    def transform(self,ex=0,ey=0):
        """Function for adding the real coordinate transfomr to the layer

        Note: for no mapping, set the width to 0
    
        Args:
            ex (float): relative width of the unmapped region in x direction. Default is 0 (no mapping)
            ey (float): relative width of the unmapped region in y direction. Default is 0 (no mapping)
        """
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


    def transform_complex(self,ex=0,ey=0):
        """Function for adding the complex coordinate transfomr to the layer

        Note: for no mapping, set the width to 0
    
        Args:
            ex (float): relative width of the unmapped region in x direction. Default is 0 (no mapping)
            ey (float): relative width of the unmapped region in y direction. Default is 0 (no mapping)
        """
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
        """Function for adding matrix of a coordinate transform

        Args:
            ex (float): relative width of the unmapped region in x direction. Default is 0. This is only for keeping track of the value, as it has no effect on the transformation.
            FX (ndarray): FX matrix of the coordinate trasnformation
            ey (float): relative width of the unmapped region in y direction. Default is 0. This is only for keeping track of the value, as it has no effect on the transformation.
            FY (ndarray): FY matrix of the coordinate trasnformation                       
        """        
        if (ex!=0):
            self.TX=True
            self.ex=ex
            self.FX=FX
        if (ey!=0):
            self.TY=True
            self.ey=ey
            self.FY=FY
        
        


    def mode(self,k0,kx=0.0,ky=0.0,v=True):
        """Calculates the eighenmode of the layer
        
        Args:
            k0 (float): Vacuum wavevector
            kx (float): Wavevector in the x direction
            ky (float): Wavevector in the y direction
        """
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
        if v:
            [self.W,self.V]=linalg.eig(self.M)
            self.gamma=np.sqrt(self.W)*np.sign(np.angle(self.W)+0.5*np.pi)
            if np.any(np.real(self.gamma)+np.imag(self.gamma)<=0.0):
                print('Warining: wrong complex root')
#            self.gamma=np.sqrt(self.W)
#            self.gamma=self.gamma*np.sign(np.imag(self.gamma))
#            if np.any(np.imag(self.gamma)<=0.0):
#                print 'Warining: negative imaginary part'
            if np.any(np.abs(self.gamma)<=0.0):
                print('Warining: gamma=0')
            self.VH=np.dot(self.GH,self.V/self.gamma)
        else:
            self.W=linalg.eigvals(self.M)


    def clear(self):
        """Removes data created in mode method
        """
        self.VH=None
        self.M=None
        self.GH=None
        self.W=None
        self.V=None
        self.gamma=None

    def slim(self):
        """Removes a lot of attributes. Not sure why was implemented
        """
        self.EPS1=None
        self.EPS2=None
        self.FOUP=None
        self.FX=None
        self.FY=None
        self.GH=None
        self.INV=None
        self.M=None

    def get_index(self,ordered=True):
        """Returns the effective idexes of the modes

        Args:
            ordered (bool): if True (default) the modes are ordered by decreasing effective index
        """
        if ordered:
            Neff=np.sort(self.gamma)[::-1]
        else:
            Neff=self.gamma
        return Neff


                
    def mat_plot(self,name,N=100,s=1):
        """Plot the absolute values of the fourier trasnsform matrices

        Args:
            name (str): name of the pdf file for plotting
            N (int): number of points for plotting the epsilon
            s (float): number pf relicas of the cell to plot. Default is 1.
        """
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
        """Plot the matrix of the eigenvalue problem

        Args:
            pdf (PdfPages): pdf object to be used to plot.

        Returns:
            None.

        """
        plt.figure()
        plt.title('k0:%5.3f kx:%5.3f ky:%5.3f' % (self.k0,self.kx,self.ky))
        plt.imshow(np.abs(np.abs(self.M)),aspect='auto',interpolation='nearest')
        plt.colorbar()
        pdf.savefig()
        plt.close()

    def plot_E(self,pdf,i,N=100,s=1,func=np.abs):
        """Plots the electric filed of a mode

        Args:
            pdf (PdfPages): pdf for saving the plots
            i (int): Numner of mode to be plotted (start from 1), only ordered
            func (callable): function to apply to the filed before plotting (default, np.abs). Usefulc could be np.real or np.imag
        """
        j=np.argsort(self.W)[-i-1]
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
        """Plots the magnetic filed of a mode

        Args:
            pdf (PdfPages): pdf for saving the plots
            i (int): Numner of mode to be plotted (start from 1), only ordered
            func (callable): function to apply to the filed before plotting (default, np.abs). Usefulc could be np.real or np.imag
        """
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


    def plot_field(self,pdf,i,N=100,s=1,func=np.abs,title=None,ordered=True):
        """Plots both electrinc and magnetic field

        Args:
            pdf (multiple, optional): Multiple choice. Each one has a diffrent meaning:
                - (PdfPages): append figure to the PdfPages object
                - (str): save the figure to a pdf with this name
            i (int): mode to be plotted.
            N (int): number of points in the graph. Default is 100
            s (float): number of replicas of the cell to be plotted. Default is 1
            func (callable): function to apply to the filed before plotting (default, np.abs). Usefulc could be np.real or np.imag
            title (str): title of the plot
            ordered (bool): if True, the modes are ordered by decreasing effective index
        """
        j=np.argsort(self.W)[-i-1] if ordered else i
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
        fig = plt.figure()
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
        if title!=None:
            plt.suptitle(title)
        sub.savefig(pdf, fig)



    def write_field(self,i,filename='field.out',N=100,s=1.0,func=None,ordered=True):
        """Writes the modal fields to a file
        
        Args:
            i (int): Index of the mode to plot. Menaning depends on the value of the ordered parameter.
            filename (str, optional): Name of the output file. Defaults to 'field.out'.
            N (int, optional): Number of points in (both in x and y) on which the fields are calculated. Defaults to 100.
            s (float, optional): Number of unit cells which are written. Defaults to 1.0.
            func (function, optional): Function to be applied to the field before writing. Defaults to None (full complex fields are plotted).
            ordered (bool, optional): if True, modes are ordered on decreasing effective index. If False, the order is whatever is provided by the diagonalization routine. Defaults to True.

        Returns:
            None.

        """
        if ordered:
            j=np.argsort(self.W)[-i-1]
        else:
            j=i
        if self.TX:
            ex=self.ex
            if s==1.0:
                x=np.linspace(-0.5,0.5,N)
                xl=sub.t_dir(x,ex)
            else:
                xl=np.linspace(-s*0.5,s*0.5,N)
                x=sub.t_inv(xl,ex)
        else:
            x=np.linspace(-0.5,0.5,N)
            xl=np.linspace(-0.5,0.5,N)

        if self.TY:
            ey=self.ey
            if s==1.0:
                y=np.linspace(-0.5,0.5,N)
                yl=sub.t_dir(y,ey)
            else:
                yl=np.linspace(-s*0.5,s*0.5,N)
                y=sub.t_inv(yl,ey)
        else:
            y=np.linspace(-0.5,0.5,N)
            yl=np.linspace(-0.5,0.5,N)

        [X,Y]=np.meshgrid(x,y,indexing='ij')
        [WEx,WEy]=np.split(self.V[:,j],2)
        [WHx,WHy]=np.split(self.VH[:,j],2)
        Ex,Ey=np.zeros((N,N),dtype='complex'),np.zeros((N,N),dtype='complex')
        Hx,Hy=np.zeros((N,N),dtype='complex'),np.zeros((N,N),dtype='complex')
        for i in range(self.D):
            EXP=np.exp((0+2j)*np.pi*((self.G[i][0]+self.kx)*X+(self.G[i][1]+self.ky)*Y))
            Ex=np.add(Ex,np.dot(WEx[i],EXP))
            Ey=np.add(Ey,np.dot(WEy[i],EXP))
            Hx=np.add(Hx,np.dot(WHx[i],EXP))
            Hy=np.add(Hy,np.dot(WHy[i],EXP))
        if func==None:
            f=open(filename,'w')
            f.write('#    x           xl          y           yl          ReEx        ImEx        ReEy        ImEy        ReHx        ImHx        ReHy        ImHy\n')
            for i in range(N):    
                for j in range(N):
                    #print 6*'%12.6f' % (X[i,j],Y[i,j],func(Ex[i,j]),func(Ey[i,j]),func(Hx[i,j]),func(Hy[i,j]))
                    f.write(12*'%18.6e' % (x[i],xl[i],y[j]*self.Nyx,yl[j]*self.Nyx,Ex[i,j].real,Ex[i,j].imag,Ey[i,j].real,Ey[i,j].imag,Hx[i,j].real,Hx[i,j].imag,Hy[i,j].real,Hy[i,j].imag) + '\n')
                #print ''
                f.write('\n')
            f.close()

        else:
            f=open(filename,'w')
            f.write('#    x           xl          y           yl          Ex          Ey          Hx          Hy \n')
            for i in range(N):    
                for j in range(N):
                    #print 6*'%12.6f' % (X[i,j],Y[i,j],func(Ex[i,j]),func(Ey[i,j]),func(Hx[i,j]),func(Hy[i,j]))
                    f.write(8*'%18.6e' % (x[i],xl[i],y[j]*self.Nyx,yl[j]*self.Nyx,func(Ex[i,j]),func(Ey[i,j]),func(Hx[i,j]),func(Hy[i,j])) + '\n')
                #print ''
                f.write('\n')
            f.close()

    def writeE(self,i,filename='fieldE.out',N=100):
        """Writes the modal electric field to a file
        
        Args:
            i (int): Index of the mode. Modes are ordered by default.
            filename (str, optional): Name of the output file. Defaults to 'fieldE.out'.
            N (int, optional): Number of points in (both in x and y) on which the fields are calculated. Defaults to 100.

        Returns:
            None.

        """
        j=np.argsort(self.W)[-i]
        [X,Y]=np.meshgrid(np.linspace(-0.5,0.5,N),np.linspace(-0.5,0.5,N))
        [WEx,WEy]=np.split(self.V[:,j],2)
        [WHx,WHy]=np.split(self.VH[:,j],2)
        Ex,Ey=np.zeros((N,N),dtype='complex'),np.zeros((N,N),dtype='complex')
        for i in range(self.D):
            EXP=np.exp((0+2j)*np.pi*((self.G[i][0]+self.kx)*X+(self.G[i][1]+self.ky)*Y))
            Ex=np.add(Ex,np.dot(WEx[i],EXP))
            Ey=np.add(Ey,np.dot(WEy[i],EXP))
        f=open(filename,'w')
        f.write('#    x           y          ReEx        ImEx        AbsEx       ReEy        ImEy        AbsEy \n')
        for i in range(N):    
            for j in range(N):
                f.write(8*'%12.6f' % (X[i,j],Y[i,j],Ex[i,j].real,Ex[i,j].imag,abs(Ex[i,j]),Ey[i,j].real,Ey[i,j].imag,abs(Ey[i,j])) + '\n')
            f.write('\n')
        f.close()

    def writeH(self,i,filename='fieldH.out',N=100):
        """Writes the modal magnetic field to a file
        
        Args:
            i (int): Index of the mode. Modes are ordered by default.
            filename (str, optional): Name of the output file. Defaults to 'fieldH.out'.
            N (int, optional): Number of points in (both in x and y) on which the fields are calculated. Defaults to 100.

        Returns:
            None.

        """
        j=np.argsort(self.W)[-i]
        [X,Y]=np.meshgrid(np.linspace(-0.5,0.5,N),np.linspace(-0.5,0.5,N))
        [WEx,WEy]=np.split(self.V[:,j],2)
        [WHx,WHy]=np.split(self.VH[:,j],2)
        Hx,Hy=np.zeros((N,N),dtype='complex'),np.zeros((N,N),dtype='complex')
        for i in range(self.D):
            EXP=np.exp((0+2j)*np.pi*((self.G[i][0]+self.kx)*X+(self.G[i][1]+self.ky)*Y))
            Hx=np.add(Hx,np.dot(WHx[i],EXP))
            Hy=np.add(Hy,np.dot(WHy[i],EXP))
        f=open(filename,'w')
        f.write('#    x           y          ReHx        ImHx        AbsHx       ReHy        ImHy        AbsHy \n')
        for i in range(N):    
            for j in range(N):
                f.write(8*'%12.6f' % (X[i,j],Y[i,j],Hx[i,j].real,Hx[i,j].imag,abs(Hx[i,j]),Hy[i,j].real,Hy[i,j].imag,abs(Hy[i,j])) + '\n')
            f.write('\n')
        f.close()


    def write_fieldgeneral(self,u,d=None,filename='field_general.out',N=100,s=1.0):
        """Writes the electric filed obtained by a superposition on modes
        
        Args:
            u (1darray): Array of the coefficients of the forward propagating modes. Dimension in 2*number of plane waves.
            d (1darray, optional): Array of the coefficients of the backward propagating modes. Dimension in 2*number of plane waves. Defaults to None.
            filename (str, optional): Name of output file. Defaults to 'field_general.out'.
            N (int, optional):  Number of points in (both in x and y) on which the fields are calculated. Defaults to 100.
            s (float, optional): Number of unit cells which are written. Defaults to 1.0.

        Returns:
            None.

        """
        if d is None:
            d=np.zeros(2*self.D,dtype=complex)

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

        [X,Y]=np.meshgrid(x,y,indexing='ij')
        Ef=np.dot(self.V,u+d)
        [WEx,WEy]=np.split(Ef,2)
        #[WEx,WEy]=np.split(self.V[:,j],2)
        #[WHx,WHy]=np.split(self.VH[:,j],2)
        Ex,Ey=np.zeros((N,N),dtype='complex'),np.zeros((N,N),dtype='complex')
        for i in range(self.D):
            EXP=np.exp((0+2j)*np.pi*((self.G[i][0]+self.kx)*X+(self.G[i][1]+self.ky)*Y))
#            Ex+=WEx[i]*EXP
#            Ey+=WEy[i]*EXP
            Ex=np.add(Ex,np.dot(WEx[i],EXP))
            Ey=np.add(Ey,np.dot(WEy[i],EXP))
        f=open(filename,'w')
        f.write('#    x           y          ReEx        ImEx        AbsEx       ReEy        ImEy        AbsEy \n')
        for i in range(N):    
            for j in range(N):
                f.write(10*'%12.6f' % (x[i],xl[i],y[j]*self.Nyx,yl[j]*self.Nyx,Ex[i,j].real,Ex[i,j].imag,abs(Ex[i,j]),Ey[i,j].real,Ey[i,j].imag,abs(Ey[i,j])) + '\n')
            f.write('\n')
        f.close()


    def get_field(self,x,y,i,func=None):
        """Returns the modal fields at a specific point
        
        Args:
            x (float): x coordinate of requested field.
            y (float): y coordinate of requested field.
            i (int): Index of the mode. Moder are ordered by default
            func (function, optional): Function to be applied to the field before returning. Defaults to None (full complex value is returned).

        Returns:
            1darray: Array containing Ex, Ey, Hx, Hy.

        """
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
        if func is not None:
            return func(np.array([Ex,Ey,Hx,Hy]))
        else:
            return np.array([Ex,Ey,Hx,Hy])

    def get_field2(self,X,Y,i,func=None):
        """Returns the modal fields given the coordinates
        
        Args:
            X (ndarray): Array of x points (unit of ax).
            Y (ndarray): Array of y points (unit of ay).
            i (int): Index of the mode. Modes are ordered by default
            func (function, optional): Function to be applied to the field before returning. Defaults to None (full complex value is returned).

        Raises:
            ValueError: Raised if X and Y have different shapes

        Returns:
            res (tuple): tuple of ndarrays, containing Ex, Ey, Hx, Hy

        """
        if np.shape(X)!=np.shape(Y):
            raise ValueError('X and Y arrays have different shapes')
        j=np.argsort(self.W)[-i-1]
        [WEx,WEy]=np.split(self.V[:,j],2)
        [WHx,WHy]=np.split(self.VH[:,j],2)
        Ex=np.zeros_like(X,dtype=complex)
        Ey=np.zeros_like(X,dtype=complex)
        Hx=np.zeros_like(X,dtype=complex)
        Hy=np.zeros_like(X,dtype=complex)        
        for i in range(self.D):
            EXP=np.exp((0+2j)*np.pi*((self.G[i][0]+self.kx)*X+(self.G[i][1]+self.ky)*Y))
            Ex=np.add(Ex,np.dot(WEx[i],EXP))
            Ey=np.add(Ey,np.dot(WEy[i],EXP))
            Hx=np.add(Hx,np.dot(WHx[i],EXP))
            Hy=np.add(Hy,np.dot(WHy[i],EXP))
        if func is not None:
            res=(func(M) for M in [Ex,Ey,Hx,Hy])
        else:
            res = (Ex,Ey,Hx,Hy)
        return res


    def plot_Et(self,pdf,i,N=100,sx=1,sy=1,func=np.abs):
        """Plots both Ex and Ey

        Args:
            pdf (multiple, optional): Multiple choice. Each one has a diffrent meaning:
                - (PdfPages): append figure to the PdfPages object
                - (str): save the figure to a pdf with this name
            i (int): Modal index. Modes are ordered by default.
            N (int, optional): DESCRIPTION. Defaults to 100.
            sx (float, optional): Width of the plotting windows in x (unit of ax). Defaults to 1.
            sy (float, optional): Width of the plotting windows in y (unit of ay). Defaults to 1.
            func (TYPE, optional): Function to apply to the field before plotting. Defaults to np.abs.

        Returns:
            None.

        """
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
        """Creates array of single mode Poynting vector components.
        
        It is stored in the P_norm attribute
        
        Returns:
            None.

        """
        [VEx,VEy]=np.split(self.V,2)
        [VHx,VHy]=np.split(self.VH,2)
        self.P_norm=np.sum(VEx*np.conj(VHy)-VEy*np.conj(VHx),0).real

    def get_Poynting_single(self,i,u,ordered=True):
        """Return the Poyinting vector of a single mode given the modal expansion in the layer
        

        Args:
            i (int): Index of the mode.
            u (1darray): Array of modal coefficient.
            ordered (TYPE, optional): Regulates how mode are ordered. If True, they are ordered for decreasing effective index. If Flase, the order is whatever is returned by the diagonalization routine. Defaults to True.

        Returns:
            TYPE: DESCRIPTION.

        """
        if ordered:
            j=np.argsort(self.W)[-i-1]
        else:
            j=i
        self.get_Poyinting_norm()
        return self.PP_norm[j,j].real*np.abs(u[j])**2.0
        #self.get_P_norm()
        #return self.P_norm[j].real*np.abs(u[j])**2.0

   
    def get_Poyinting_norm(self):
        """Calculates the normalization matrix for the Poyinting vector calculations
        

        Returns:
            None.

        """
        [VEx,VEy]=np.split(self.V,2)
        [VHx,VHy]=np.conj(np.split(self.VH,2))        
        #old version (working) 
        #self.PP_norm=np.zeros((2*self.D,2*self.D),dtype=complex)
        #for i in range(self.D):
        #    VEX,VHY=np.meshgrid(VEx[i,:],VHy[i,:])
        #    VEY,VHX=np.meshgrid(VEy[i,:],VHx[i,:])
        #    P1=np.multiply(VEX,VHY)
        #    P2=-np.multiply(VEY,VHX)
        #    P=np.add(P1,P2)
        #    self.PP_norm=np.add(self.PP_norm,P)
        #print self.PP_norm
        #new version. should be equivalent bit faster
        P1=np.dot(np.transpose(VEx),VHy)        
        P2=np.dot(np.transpose(VEy),VHx)
        self.PP_norm=np.add(P1,-P2)

    
    def get_Poynting(self,u,d=None):
        """Calculates total Poynting vector in the layer given arrays of modal expansion
        

        Args:
            u (1darray): Modal expansion of forward propagating modes.
            d (1darray, optional): Modal expansion of backward propagating modes. Defaults to None.

        Returns:
            TYPE: DESCRIPTION.

        """
        if d is None:
            d=np.zeros(2*self.D,dtype=complex)
        #try:
        #    self.PP_norm
        #except AttributeError:
        #    self.get_Poyinting_norm()
        self.get_Poyinting_norm()
        Cn=np.add(u,d)
        Cnp=np.add(u,-d)
        [Cn,Cnp]=np.meshgrid(Cn,np.conj(Cnp))
        C=np.multiply(Cn,Cnp)
        PP=np.multiply(C,self.PP_norm)
        return np.sum(PP).real


    def T_interface(self,lay):
        """Builds the Transfer matrix if the interface with another layer
        
        Args:
            lay (Layer): Layer toward which to calculate the scattering matrix.

        Returns:
            T (2darray): Interface scattering matrix.

        """
        T1=np.dot(linalg.inv(lay.V),self.V)
        T2=np.dot(linalg.inv(lay.VH),self.VH)
        T11= 0.5*(T1 + T2)
        T12= 0.5*(T1 - T2)
        T21= 0.5*(T1 - T2)
        T22= 0.5*(T1 + T2)
        T=np.vstack([np.hstack([T11,T12]),np.hstack([T21,T22])])
        return T

    def T_prop(self,d):
        """Build the propagation Transfer matrix of the layer
        

        Args:
            d (float): Thickness of the layer.

        Returns:
            T (2darray): Propagation Transfer matrix.

        """
        I1=np.diag(np.exp((0+1j)*self.k0*self.gamma*d))
        I2=np.diag(np.exp(-(0+1j)*self.k0*self.gamma*d))
        I=np.zeros((2*self.D,2*self.D),complex)
        T=np.vstack([np.hstack([I1,I]),np.hstack([I,I2])])
        return T
        
#newer version, should be faster
    def interface(self,lay):
        """Builds the Scattering matrix if the interface with another layer
        
        Args:
            lay (Layer): Layer toward which to calculate the scattering matrix.

        Returns:
            S (S_matrix): Interface scattering matrix.

        """
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


    def get_input(self,func,args=None,Nxp=1024,Nyp=None,fileprint=None):
        """Expands an arbitrary fieldd shape on the basis of the layer eigenmodes
        

        Args:
            func (function): Function describing the field.
                This function should be in the form (x,y,*args). It must be able to accept x and y as numpy array.
                It must return two values, expressing Ex and Ey
            args (tuple, optional): Eventual tuple of additional arguments for func. Defaults to None.
            Nxp (int, optional): Number of points to evaluate the function in the x direction. Defaults to 1024.
            Nyp (int, optional): Number of points to evaluate the function in the y direction. Defaults to None (1 if layer is 1D, Nxp if 2D).
            fileprint (str, optional): Filename on which to write the used function. Mainly for debug. Defaults to None.

        Returns:
            u (1darray): Array of the modal coefficient of the expansion.

        """
        args = () if args is None else args
        
        if Nyp==None:
            if self.Ny==0:
                Nyp=1
                y=np.array([0.0])
            else:
                Nyp=Nxp
                y=np.linspace(-0.5,0.5,Nyp)
        else:
            y=np.linspace(-0.5,0.5,Nyp)
        x=np.linspace(-0.5,0.5,Nxp)
        if self.TX:
            ex=self.ex
            x=sub.t_dir(x,ex)
        if self.TY:
            ey=self.ey
            y=sub.t_dir(y,ey)

        y=y*self.Nyx

        [X,Y]=np.meshgrid(x,y,indexing='ij')
        [Fx,Fy]=func(X,Y,*args)

        try:
            f=open(fileprint,'w')
            for i in range(Nxp):
                for j in range(Nyp):
                    f.write(6*'%18.8e' % (x[i],y[j],Fx[i,j].real,Fx[i,j].imag,Fy[i,j].real,Fy[i,j].imag))
                    f.write('\n')
                f.write('\n')
        except TypeError:
            pass

        Fx=np.fft.fftshift(Fx)/(Nxp*Nyp)
        Fy=np.fft.fftshift(Fy)/(Nxp*Nyp)

        FOUx=np.fft.fft2(Fx)
        FOUy=np.fft.fft2(Fy)

        Estar=np.zeros(2*self.NPW,dtype=complex)
        for i in range(self.NPW):
            #print self.G[i][0], self.G[i][1],FOUx[self.G[i][0],self.G[i][1]]              
            Estar[i]=FOUx[self.G[i][0],self.G[i][1]]
            Estar[i+self.NPW]=FOUy[self.G[i][0],self.G[i][1]]

        #for i in range(-self.Nx,self.Nx+1):
        #    for j in range(-self.Ny,self.Ny+1):
        #        print '%4i %4i %15.8e %15.8e' % (i,j,np.abs(FOUx[i,j]),np.abs(FOUy[i,j]))
        #    print ''

        u=linalg.solve(self.V,Estar)
        return u

    def create_input(self,dic):
        """Creates the array of modal coefficient using a dictionary as input
        

        Args:
            dic (dict): Dictionary of exited modes {modal_index : modal_coeff}. Modes are ordered. 

        Returns:
            u (1darray): Array of modal coefficient.

        """
        u=np.zeros((2*self.NPW),complex)
        for i in dic:
            u[np.argsort(self.W)[-i]]=dic[i]
        return u

    def get_Enorm(self):
        """Calculate field normalization
        

        Returns:
            None.

        """
        [VEx,VEy]=np.split(self.V,2)
        self.ENx=np.dot(np.transpose(VEx),np.conj(VEx))
        self.ENy=np.dot(np.transpose(VEy),np.conj(VEy))

    def overlap(self,u,up=None):
        """EXPERIMENTAL: Calculates overlap between two fields given the modal expansion
        

        Args:
            u (1darray): Modal coefficient of first mode.
            up (1darray, optional): Modal coefficient of first mode. Defaults to None (up=u, namely normalization is returned).

        Returns:
            list: [tx, tx]: floats. Namely overlap in x and y polarization

        """
        if up is None:
           up=u
        try:
            self.ENx
        except AttributeError:
            self.get_Enorm()
        #print np.shape(u),np.shape(self.ENx)
        tx=np.dot(self.ENx,np.conj(up))
        ty=np.dot(self.ENy,np.conj(up))
        tx=np.dot(np.transpose(u),tx)
        ty=np.dot(np.transpose(u),ty)
        #print tx,ty
        return [tx,ty]


    def coupling(self,u,up):
        """EXPERIMENTAL: Calculates coupling between two modes given their modal exapnsion
        

        Args:
            u (TYPE): Modal coefficient of first mode.
            up (TYPE): Modal coefficient of second mode.

        Returns:
            list: [tx, tx]: floats. Coupling in x and y polarization.

        """
        self.get_Enorm()
        [tx1,ty1]=self.overlap(u)
        [tx2,ty2]=self.overlap(up)
        [txc,tyc]=self.overlap(u,up)
        return [txc/np.sqrt(tx1*tx2),tyc/np.sqrt(ty1*ty2)]


class Layer_ani_diag(Layer):
    """ Class for the definition of a single layer anysitropic (diagonal) layer
    """
    def __init__(self,Nx,Ny,creator_x,creator_y,creator_z,Nyx=1.0):
        self.Nx=Nx
        self.Ny=Ny
        self.NPW=(2*Nx+1)*(2*Ny+1)
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


class Layer_num(Layer):
    """ Class for the definition of a single layer from a function defining the dielectric profile
    """
    def __init__(self,Nx,Ny,func,args=(),Nyx=1.0,NX=2048,NY=2048):
        self.Nx=Nx
        self.Ny=Ny
        self.NPW=(2*Nx+1)*(2*Ny+1)
        self.G=sub.createG(self.Nx,self.Ny)
        self.D=len(self.G)
        self.Nyx=Nyx
        self.func=func
        self.args=args
        self.NX=NX
        self.NY=NY
        #print args
        
        self.FOUP=sub.num_fou(func,args,self.G,NX,NY,self.Nyx)
        self.INV=linalg.inv(self.FOUP)

        #Still to be defined
        self.EPS1=sub.num_fou_xy(self.func,self.args,self.Nx,self.Ny,self.G,NX,NY,self.Nyx)
        self.EPS2=sub.num_fou_yx(self.func,self.args,self.Nx,self.Ny,self.G,NX,NY,self.Nyx)
        

        self.TX=False
        self.TY=False


    def eps_plot(self,pdf=None,N=200,s=1.0):
        """Function for plotting the dielectric consstat rebuit from plane wave expansion

        Args:
            pdf (string or PdfPages): file for printing the the epsilon
                if a PdfPages object, the page is appended to the pdf
                if string, a pdf with that name is created
            N (int): number of points
            s (float): number of cell replicas to display (default 1)
        """
        [X,Y]=np.meshgrid(np.linspace(-s*0.5,s*0.5,s*N),np.linspace(-s*0.5,s*0.5,s*N))
        [YY,XX]=np.meshgrid(np.linspace(-0.5,0.5,self.NY),np.linspace(-0.5,0.5,self.NX))
        #F=self.func(XX,YY)
        F=self.func(XX,YY/self.Nyx,*self.args)
        F=np.fft.fftshift(F)
        FOU=np.fft.fft2(F)/self.NX/self.NY
        EPS=np.zeros((N,N),complex)
        EPS=sum([FOU[self.G[i][0],self.G[i][1]]*np.exp((0+2j)*np.pi*(self.G[i][0]*X+self.G[i][1]*Y)) for i in range(self.D)])
        plt.figure()
        plt.imshow(np.real(EPS),origin='lower',extent=[-s*0.5,s*0.5,-self.Nyx*s*0.5,self.Nyx*s*0.5])
        plt.colorbar()
        if pdf is None:
            pass
        elif isinstance(pdf,PdfPages):
            pdf.savefig()
        else:
            a=PdfPages(pdf+'.pdf')
            a.savefig()
            a.close()
        plt.close()


    def mat_plot(self,name,N=100,s=1):
        """Plot the absolute values of the fourier trasnsform matrices

        Args:
            name (str): name of the pdf file for plotting
            N (int): number of points for plotting the epsilon
            s (float): number pf relicas of the cell to plot. Default is 1.
        """
        save=PdfPages(name+'.pdf')

        [X,Y]=np.meshgrid(np.linspace(-s*0.5,s*0.5,s*N),np.linspace(-s*0.5,s*0.5,s*N))
        [YY,XX]=np.meshgrid(np.linspace(-0.5,0.5,self.NY),np.linspace(-0.5,0.5,self.NX))
        #F=self.func(XX,YY)
        F=self.func(XX,YY/self.Nyx,*self.args)
        F=np.fft.fftshift(F)
        FOU=np.fft.fft2(F)/self.NX/self.NY
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


class Layer_uniform(Layer):
    """ Class for the definition of a single uniform layer
    """
    def __init__(self,Nx,Ny,eps,Nyx=1.0):
        self.Nx=Nx
        self.Ny=Ny
        self.NPW=(2*Nx+1)*(2*Ny+1)
        self.G=sub.createG(self.Nx,self.Ny)
        self.D=len(self.G)
        self.Nyx=Nyx
        self.eps=eps
        
        self.FOUP=eps*np.identity(self.D,dtype='complex')
        self.INV=1.0/eps*np.identity(self.D,dtype='complex')

        #Still to be defined
        self.EPS1=eps*np.identity(self.D,dtype='complex')
        self.EPS2=eps*np.identity(self.D,dtype='complex')

        self.TX=False
        self.TY=False


    def eps_plot(self,pdf=None,N=200,s=1.0):
        """Function for plotting the dielectric consstat rebuit from plane wave expansion

        Args:
            pdf (string or PdfPages): file for printing the the epsilon
                if a PdfPages object, the page is appended to the pdf
                if string, a pdf with that name is created
            N (int): number of points
            s (float): number of cell replicas to display (default 1)
        """
        [X,Y]=np.meshgrid(np.linspace(-s*0.5,s*0.5,s*N),np.linspace(-s*0.5,s*0.5,s*N))
        EPS=np.zeros(np.shape(X))+self.eps.real
        plt.figure()
        plt.imshow(np.real(EPS),extent=[-s*0.5,s*0.5,-self.Nyx*s*0.5,self.Nyx*s*0.5],origin='lower')
        plt.colorbar()
        if (pdf==None):
            plt.show()
        else:
            a=PdfPages(pdf+'.pdf')
            a.savefig()
            a.close()
        plt.close()


    def mat_plot(self,name,N=100,s=1):
        """Plot the absolute values of the fourier trasnsform matrices

        Args:
            name (str): name of the pdf file for plotting
            N (int): number of points for plotting the epsilon
            s (float): number pf relicas of the cell to plot. Default is 1.
        """
        save=PdfPages(name+'.pdf')

        [X,Y]=np.meshgrid(np.linspace(-s*0.5,s*0.5,s*N),np.linspace(-s*0.5,s*0.5,s*N))
        #F=self.func(XX,YY)
        EPS=np.zeros(np.shape(X))+self.eps.real

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


    def mode(self,k0,kx=0.0,ky=0.0):
        """Calculates the eighenmode of the layer
        
        Args:
            k0 (float): Vacuum wavevector
            kx (float): Wavevector in the x direction
            ky (float): Wavevector in the y direction
        """
        self.k0=k0
        self.kx=kx
        self.ky=ky
        if (self.TX or self.TY):
            (k1,k2)=sub.createK(self.G,k0,kx=kx,ky=ky,Nyx=self.Nyx)
            if self.TX:
                #print np.shape(self.FX),np.shape(self.k1)
                k1=np.dot(self.FX,k1)
            if self.TY:
                k2=np.dot(self.FY,k2)
            self.GH,self.M=sub.create_2order_new(self.D,k1,k2,self.INV,self.EPS1,self.EPS2)
            [self.W,self.V]=linalg.eig(self.M)
            self.gamma=np.sqrt(self.W)*np.sign(np.angle(self.W)+0.5*np.pi)
            if np.any(np.real(self.gamma)+np.imag(self.gamma)<=0.0):
                print('Warining: wrong complex root')
            if np.any(np.abs(self.gamma)<=0.0):
                print('Warining: gamma=0')
            self.VH=np.dot(self.GH,self.V/self.gamma)
        else:
            W=2*[self.eps-((1+0j)*(self.G[i][0]+kx)/k0)**2-((1+0j)*(self.G[i][1]+ky)/k0/self.Nyx)**2 for i in self.G]
            self.W=np.array(W)
            self.V=np.identity(2*self.D,dtype=complex)
            self.gamma=np.sqrt(self.W)*np.sign(np.angle(self.W)+0.5*np.pi)
            if np.any(np.real(self.gamma)+np.imag(self.gamma)<=0.0):
                print('Warining: wrong complex root')
            if np.any(np.abs(self.gamma)<=0.0):
                print('Warining: gamma=0')       
            self.M=np.diag(self.V)
            GH_11=[-(1+0j)*(self.G[i][1]+ky)/k0/self.Nyx*(self.G[i][0]+kx)/k0 for i in self.G]
            GH_22=[(1+0j)*(self.G[i][1]+ky)/k0/self.Nyx*(self.G[i][0]+kx)/k0 for i in self.G]
            GH_12=[((1+0j)*(self.G[i][0]+kx)/k0)**2-self.eps for i in self.G]
            GH_21=[self.eps-((1+0j)*(self.G[i][1]+ky)/k0/self.Nyx)**2 for i in self.G]
            self.GH=np.vstack([np.hstack([np.diag(GH_11),np.diag(GH_12)]),np.hstack([np.diag(GH_21),np.diag(GH_22)])])
            self.VH=np.dot(self.GH,self.V/self.gamma)



class Layer_empty_st(Layer):
    """Class for the definition of an empy layer
    """
    def __init__(self,Nx,Ny,creator,Nyx=1.0):
        self.Nx=Nx
        self.Ny=Ny
        self.NPW=(2*Nx+1)*(2*Ny+1)
        self.G=sub.createG(self.Nx,self.Ny)
        self.D=len(self.G)
        self.creator=copy.deepcopy(creator)
        self.Nyx=Nyx

        self.TX=False
        self.TY=False

        self.FOUP=np.zeros((self.D,self.D),dtype=complex)
        #self.INV=np.zeros((self.D,self.D),dtype=complex)
        self.INV=linalg.inv(np.eye(self.D,dtype=complex))
        self.EPS1=np.zeros((self.D,self.D),dtype=complex)
        self.EPS2=np.zeros((self.D,self.D),dtype=complex)

    def fourier(self):
        """Calculates the fourier transform matrices need for the eigenvalue problem.
        

        Returns:
            2darray: FOUP matrix.
            2darray: INV matrix.
            2darray: EPS1 matrix.
            2darray: EPS2 matrix.

        """
        self.FOUP=sub.create_epsilon(self.G,self.creator.x_list,self.creator.y_list,self.creator.eps_lists)*(1.0+0.0j)
        self.INV=linalg.inv(self.FOUP)
        self.EPS1=sub.fou_xy(self.Nx,self.Ny,self.G,self.creator.x_list,self.creator.y_list,self.creator.eps_lists)*(1.0+0.0j)
        self.EPS2=sub.fou_yx(self.Nx,self.Ny,self.G,self.creator.x_list,self.creator.y_list,self.creator.eps_lists)*(1.0+0.0j)
        return (self.FOUP,self.INV,self.EPS1,self.EPS2)


class Layer_from_xsection(Layer):  
    """ Development class
    """
    def __init__(self,Nx,Ny,xs):
        self.Nx=Nx
        self.Ny=Ny
        self.NPW=(2*Nx+1)*(2*Ny+1)
        self.G=sub.createG(self.Nx,self.Ny)
        self.D=len(self.G)
        self.creator=Creator()

        y_stacks=[]
        w_tot=0.0
        x_list=[]
        for i, (width,stack) in enumerate(zip(xs.hstack['widths'],xs.hstack['vstacks'])):
            y_temp=[]
            x_list.append(width)
            w_tot+=width
            for j, (m, w) in enumerate(stack.layers):
                y_temp.append(w)
            y_temp=np.cumsum(y_temp)
            for y in y_temp:
                if not np.isclose(y_stacks,y).any():
                    y_stacks.append(y) 
        
        x_list=(np.cumsum(np.array(x_list)))/w_tot-0.5
        y_list=np.sort(y_stacks)
        h_tot=y_list[-1]     
        y_norm=[y/h_tot-0.5 for y in y_list]

        #print(y_list,'\n')
        eps_lists=[]
        for i, (width,stack) in enumerate(zip(xs.hstack['widths'],xs.hstack['vstacks'])):
            eps_list=[]
            ys=list(np.cumsum([w for (m, w) in stack.layers]))
            ys.append(h_tot) 
            inds=[m.Nmat() for (m, w) in stack.layers]
            inds.append(xs.background.Nmat())
            for y in y_list:
                for j,yp in enumerate(ys):
                    if y<=yp:
                        break
                eps_list.append(inds[j]**2.0)
            #print(ys,inds,eps_list,'\n')
            eps_lists.append(eps_list)

        self.FOUP=sub.create_epsilon(self.G,x_list,y_norm,eps_lists)
        self.INV=linalg.inv(self.FOUP)
        self.EPS1=sub.fou_xy(self.Nx,self.Ny,self.G,x_list,y_norm,eps_lists)
        self.EPS2=sub.fou_yx(self.Nx,self.Ny,self.G,x_list,y_norm,eps_lists)

        self.creator.x_list=x_list
        self.creator.y_list=y_norm
        self.creator.eps_lists=eps_lists

        #print(x_list)
        #print(y_norm)
        #print(eps_lists)


        self.TX=False
        self.TY=False
        self.Nyx=h_tot/w_tot
        self.ax=w_tot
        self.ay=h_tot

    def mode_from_lam(self,lam,kx=0.0,ky=0.0,v=1):
        self.mode(self.ax/lam,kx=kx,ky=ky,v=v)


class Layer_from_hstack(Layer):    
    """ Development class
    """
    def __init__(self,Nx,Ny,hstack):
        self.Nx=Nx
        self.Ny=Ny
        self.NPW=(2*Nx+1)*(2*Ny+1)
        self.G=sub.createG(self.Nx,self.Ny)
        self.D=len(self.G)
        self.creator=Creator()

        y_stacks=[]
        w_tot=0.0
        x_list=[]
        for i, (stack,width) in enumerate(hstack.layers):
            y_temp=[]
            x_list.append(width)
            w_tot+=width
            for j, (m, w) in enumerate(stack.layers):
                y_temp.append(w)
            if i==0:
                background=m.Neff()
            y_temp=np.cumsum(y_temp)
            for y in y_temp:
                if not np.isclose(y_stacks,y).any():
                    y_stacks.append(y) 
        
        x_list=(np.cumsum(np.array(x_list)))/w_tot-0.5
        y_list=np.sort(y_stacks)
        h_tot=y_list[-1]     
        y_norm=[y/h_tot-0.5 for y in y_list]

        #print(y_list,'\n')
        eps_lists=[]
        for i, (stack,width) in enumerate(hstack.layers):
            eps_list=[]
            ys=list(np.cumsum([w for (m, w) in stack.layers]))
            ys.append(h_tot) 
            inds=[m.Nmat() for (m, w) in stack.layers]
            inds.append(inds[-1])
            for y in y_list:
                for j,yp in enumerate(ys):
                    if y<=yp:
                        break
                eps_list.append(inds[j]**2.0)
            #print(ys,inds,eps_list,'\n')
            eps_lists.append(eps_list)

        self.FOUP=sub.create_epsilon(self.G,x_list,y_norm,eps_lists)
        self.INV=linalg.inv(self.FOUP)
        self.EPS1=sub.fou_xy(self.Nx,self.Ny,self.G,x_list,y_norm,eps_lists)
        self.EPS2=sub.fou_yx(self.Nx,self.Ny,self.G,x_list,y_norm,eps_lists)

        self.creator.x_list=x_list
        self.creator.y_list=y_norm
        self.creator.eps_lists=eps_lists

        #print(x_list)
        #print(y_norm)
        #print(eps_lists)


        self.TX=False
        self.TY=False
        self.Nyx=h_tot/w_tot
        self.ax=w_tot
        self.ay=h_tot

    def mode_from_lam(self,lam,kx=0.0,ky=0.0,v=1):
        self.mode(self.ax/lam,kx=kx,ky=ky,v=v)
        

