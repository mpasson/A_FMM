import numpy as np
import scipy.linalg as linalg

class S_matrix:
    """Implementation of the scattring matrix object
    
    This object is a container for NxN matrices, conventionally defined as S11, S12, S21 and S22
    Also, it implementens all the methods involving operations on scattring matrix
    """

    def __init__(self,N):
        """Creator
        
        Args:
            N (int): Dimension of each of the NxN submatrices of the scattring matrix. The total matrix is 2Nx2N

        Returns:
            None.

        """
        self.N=N
        self.S11=np.identity(N,complex)
        self.S22=np.identity(N,complex)
        self.S12=np.zeros((N,N),complex)
        self.S21=np.zeros((N,N),complex)


    #OLD RECURSION VERSION
    #def add(self,s):
    #    T1=np.dot(linalg.inv(np.identity(self.N,complex)-np.dot(self.S12,s.S21)),self.S11)
    #    T2=np.dot(linalg.inv(np.identity(self.N,complex)-np.dot(s.S21,self.S12)),s.S22)
    #    self.S11=np.dot(s.S11,T1)
    #    self.S12=s.S12+np.dot(np.dot(s.S11,self.S12),T2)
    #    self.S21=self.S21+np.dot(np.dot(self.S22,s.S21),T1)
    #    self.S22=np.dot(self.S22,T2)

    #NEW RECURSION VERSION
    def add(self,s):
        """Recursion method for joining two scattering matrices
        
        The connection is between the "right" side of self and the "left" side of s
        
        Args:
            s (S_matrix): scattering matrix to be joined to self. The 

        Returns:
            None.

        """
        I=np.identity(self.N,complex)
        T1=np.dot(s.S11,linalg.inv(I-np.dot(self.S12,s.S21)))
        T2=np.dot(self.S22,linalg.inv(I-np.dot(s.S21,self.S12)))
        self.S21=self.S21+np.dot(np.dot(T2,s.S21),self.S11)
        self.S11=np.dot(T1,self.S11)
        self.S12=s.S12   +np.dot(np.dot(T1,self.S12),s.S22)             
        self.S22=np.dot(T2,s.S22)

    def add_left(self,s):
        """Recursion method for joining two scattering matrices
        
        The connection is between the "left" side of self and the "right" side of s
        
        Args:
            s (S_matrix): scattering matrix to be joined to self. The 

        Returns:
            None.

        """
        T1=np.dot(linalg.inv(np.identity(self.N,complex)-np.dot(s.S12,self.S21)),s.S11)
        T2=np.dot(linalg.inv(np.identity(self.N,complex)-np.dot(self.S12,s.S21)),self.S22)
        s.S11=np.dot(self.S11,T1)
        s.S12=self.S12+np.dot(np.dot(self.S11,s.S12),T2)
        s.S21=s.S21+np.dot(np.dot(s.S22,self.S21),T1)
        s.S22=np.dot(s.S22,T2)

    def add_uniform(self,lay,d):
        """Recursion method for addig to self the progation matrix of a given layer
        
        The connection is between the "right" side of self and the "left" side of the propagation matrix  

        Args:
            lay (Layer): Layer of which to calculate the propagation matrix
            d (float): Thickness of the layer

        Returns:
            None.

        """
        E=np.diag(np.exp((0+2j)*np.pi*lay.k0*lay.gamma*d))
        self.S11=np.dot(E,self.S11)
        self.S12=np.dot(E,np.dot(self.S12,E))
        self.S22=np.dot(self.S22,E)

    def add_uniform_left(self,lay,d):
        """Recursion method for addig to self the progation matrix of a given layer
        
        The connection is between the "left" side of self and the "right" side of the propagation matrix  

        Args:
            lay (Layer): Layer of which to calculate the propagation matrix
            d (float): Thickness of the layer

        Returns:
            None.

        """
        E=np.diag(np.exp((0+2j)*np.pi*lay.k0*lay.gamma*d))
        self.S11=np.dot(self.S11,E)
        self.S21=np.dot(E,np.dot(self.S21,E))
        self.S22=np.dot(E,self.S22)


    def S_print(self,i=None,j=None):
        """Function for printing the scattering matrix. 
        
        It can print both the full matrix or the 2x2 matrix between relevant modes

        Args:
            i (int, optional): index of the "left" mode. Default is None (full matrix)
            j (int, optional): index of the "right" mode. Default is None (full matrix)

        Returns:
            None.

        """
        if i==None:
            S=np.vstack([np.hstack([self.S11,self.S12]),np.hstack([self.S21,self.S22])])
        else:
            j=i if j==None else j
            S=np.vstack([np.hstack([self.S11[i,j],self.S12[i,j]]),np.hstack([self.S21[i,j],self.S22[i,j]])])
        print(S)

    def det(self):
        """Calculate the determinat of the scattering matrix

        Returns:
            float: Determinant of the scattering matrix

        """
        return linalg.det(np.vstack([np.hstack([self.S11,self.S12]),np.hstack([self.S21,self.S22])]))

    def S_modes(self):
        """Solves the eigenvalue problem of the Bloch modes of the scattring matrix
        
        Returns:
            W (1darray): arrays of the eigenvalues (complex amplitude of the mode after one period)
            V (2darray):  arrays of the eigenvectors (Bloch modes on the base of the guided mode in the first and last layer)

        """
        ID=np.identity(self.N)
        Z=np.zeros((self.N,self.N))
        S1=np.vstack([np.hstack([self.S11,Z]),np.hstack([self.S21,-ID])])
        S2=np.vstack([np.hstack([ID,-self.S12]),np.hstack([Z,-self.S22])])
        [W,V]=linalg.eig(S1,b=S2)
        return W,V

    def det_modes(self,kz,d):
        ID=np.identity(self.N)
        Z=np.zeros((self.N,self.N))
        S1=np.vstack([np.hstack([self.S11,Z]),np.hstack([self.S21,-ID])])
        S2=np.vstack([np.hstack([ID,-self.S12]),np.hstack([Z,-self.S22])])
        return linalg.det(S1-np.exp((0.0+1.0j)*kz*d)*S2)        

    def der(self,Sm,Sp,h=0.01):
        """Calculates the first and second derivative of the scattering matrix with respec to the parameter par.

        Args:
            Sm (S_matrix): S matrix calculated at par=par0-h
            Sp (S_matrix): S matrix calculated at par=par0+h
            h (float, optional): Interval used to calculate the derivatives . Defaults to 0.01.

        Returns:
            tuple: tuple containing:
                
                - S1 (2darray): First derivative of the scattering matrix with respect to par.
                - S2 (2darray): Second derivative of the scattering matrix with respect to par.

        """
        S=np.vstack([np.hstack([self.S11,self.S12]),np.hstack([self.S21,self.S22])])
        S_m=np.vstack([np.hstack([Sm.S11,Sm.S12]),np.hstack([Sm.S21,Sm.S22])])
        S_p=np.vstack([np.hstack([Sp.S11,Sp.S12]),np.hstack([Sp.S21,Sp.S22])])
        S1=(S_p-S_m)/(2.0*h)
        S2=(S_p+S_m-2.0*S)/(h*h)
        return (S1,S2)

    def matrix(self):
        """Returns the full scattering matrix
        
        Returns:
            2darray: Scattering matrix as numpy array

        """
        return np.vstack([np.hstack([self.S11,self.S12]),np.hstack([self.S21,self.S22])])

    def output(self,u1,d2):
        """Returs the output vectors given the input vectors

        Args:
            u1 (1darray): Array of modal coefficient of "left" inputs.
            d2 (1darray): Array of modal coefficient of "right" inputs.

        Returns:
            tuple: tuple containing:
                
                - u2 (1darray): Array of modal coefficient of "right" outputs.
                - d1 (1darray): Array of modal coefficient of "left" outputs.

        """
        u2=np.add(np.dot(self.S11,u1),np.dot(self.S12,d2))
        d1=np.add(np.dot(self.S21,u1),np.dot(self.S22,d2))
        return (u2,d1)

    def left(self,u1,d1):
        """Return the "right" inout and output vectors given the "left" ones
        

        Args:
            u1 (1darray): Array of modal coefficient of "left" inputs.
            d1 (1darray): Array of modal coefficient of "left" outputs.

        Returns:
            tuple: tuple containing:
                
                - u2 (1darray): Array of modal coefficient of "right" outputs.
                - d2 (1darray): Array of modal coefficient of "right" inputs.

        """
        d2=linalg.solve(self.S22,d1-np.dot(self.S21,u1))
        u2=np.add(np.dot(self.S11,u1),np.dot(self.S21,d2))
        return (u2,d2)

    def int_f(self,S2,u):
        """Retirn the modal coefficient between two scattering matrces (self and S2)

        Args:
            S2 (S_matrix): Scattering matrix to between self and the end of the structure
            u (1darray): Array of modal coefficient of "left" inputs to self.

        Returns:
            tuple: tuple containing:
                
                - uo (TYPE): Array of coefficients of left-propagating modes in the middle 
                - do (TYPE): Array of coefficients of right-propagating modes in the middle

        """
        ID=np.identity(self.N)
        ut=np.dot(self.S11,u)
        uo=linalg.solve(ID-np.dot(self.S12,S2.S21),ut)
        do=linalg.solve(ID-np.dot(S2.S21,self.S12),np.dot(S2.S21,ut))
        return (uo,do)

    def int_f_tot(self,S2,u,d):
        """Retirn the modal coefficient between two scattering matrces (self and S2)

        Args:
            S2 (S_matrix): Scattering matrix to between self and the end of the structure
            u (1darray): Array of modal coefficient of "left" inputs to self.
            d (1darray): Array of modal coefficient of "right" inputs to S2

        Returns:
            tuple: tuple containing:
                
                - uo (TYPE): Array of coefficients of left-propagating modes in the middle 
                - do (TYPE): Array of coefficients of right-propagating modes in the middle

        """
        ID=np.identity(self.N)
        ut=np.dot(self.S11,u)
        dt=np.dot(S2.S22,d)
        uo=linalg.solve(ID-np.dot(self.S12,S2.S21),np.add(ut,np.dot(self.S12,dt)))
        do=linalg.solve(ID-np.dot(S2.S21,self.S12),np.add(np.dot(S2.S21,ut),dt))
        return (uo,do)

    int_complete = int_f_tot
            




