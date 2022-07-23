import numpy as np
import A_FMM.sub_sm as sub
from A_FMM.layer import Layer
from A_FMM.scattering import S_matrix
import matplotlib.pyplot as plt
import copy
from A_FMM.layer import Layer_empty_st
from matplotlib.backends.backend_pdf import PdfPages

# =============================================================================
# try:
#     from multiprocessing import Pool
# except ModuleNotFoundError:
#     print('WARNING: multiprocessing not available')
# =============================================================================


class Stack:
    """Class representing the multylayer object

    This class is used for the definition of the multilayer object to be simulated using fourier expansion (x and y axis) and scattering matrix algorithm (z axis).
    It is built from a list of layers and thicknesses. The value of the thickness of the first and last layer is irrelevant for the simulation, and it is used only to set the plotting window.
    """

    def __init__(self, layers: list[Layer] = None, d: list[float] = None):
        """Creator

        Args:
            layers (list, optional): List of Layers: layers of the multylayer. Defaults to None (empty list).
            d (list, optional): List of float: thicknesses of the multylayer. Defaults to None (empty list).

        Raises:
            ValueError: Raised if d and mat have different lengths

        Returns:
            None.

        """
        layers = [] if layers is None else layers
        d = [] if layers is None else d

        if len(layers) != len(d):
            raise ValueError("Different number of layers and thicknesses")
        self.N = len(layers)
        self.layers = layers
        self.d = d
        self.NPW = self.layers[0].D
        self.G = self.layers[0].G
        self.Nyx = self.layers[0].Nyx

        self.count_interface()

    def add_layer(self, lay, d):
        """Add a layer at the end of the multilayer


        Args:
            lay (Layer): Layer to be added.
            d (float): thickness of the layer.

        Returns:
            None.

        """
        self.layers.append(lay)
        self.d.append(d)
        self.N += 1
        self.count_interface()

    def transform(self, ex: float = 0, ey: float = 0, complex_transform: bool = False):
        """Function for adding the real coordinate transform to all layers of the stack

        Note: for no mapping, set the width to 0

        Args:
            ex (float): relative width of the unmapped region in x direction. Default is 0 (no mapping)
            ey (float): relative width of the unmapped region in y direction. Default is 0 (no mapping)
            complex_transform (bool): False for real transform (default), True for complex one.
        """
        Fx, Fy = self.layers[0].transform(
            ex=ex, ey=ey, complex_transform=complex_transform
        )
        for layer in self.layers[1:]:
            layer.add_transform_matrix(ex=ex, FX=Fx, ey=ey, FY=Fy)
        return Fx, Fy

    def mat_plot(self, N=100, s=1):
        """Call matplot on every layer in the stack. Save the results in multiple pdf files

        Args:
            N (int, optional): Number of points to be used to plot the epsilon. Defaults to 100.
            s (float, optional): Number of replicas of the unit cell to be plotted. Defaults to 1.

        Returns:
            None.

        """
        n = 1
        for lay in self.layers:
            lay.mat_plot("layer_%i" % (n), N=N, s=s)
            n += 1

    def plot_stack(self, pdf=None, N=100, dz=0.01, y=0.0, func=np.abs, cmap="viridis"):
        """Plots the stack xz cross section


        Args:
            pdf (multiple, optional): Multiple choice. Each one has a diffrent meaning:
                - (PdfPages): append figure to the PdfPages object
                - (str): save the figure to a pdf with this name
                - None (Default): Do not plot anything. Keep the figure as the active one.
            N (int, optional): Number of points in the x direction. Defaults to 100.
            dz (float, optional): Resolution in z direction (unit of ax). Defaults to 0.01.
            y (float, optional): y coordinate at which the cross section is taken (unit of ay). Defaults to 0.0.
            func (callable, optional): Function to be applied to eps befor plotting. Defaults to np.abs.
            cmap (str, optional): matplotlib colormap for plotting. Defaults to 'viridis'.

        Returns:
            None.

        """
        X = np.linspace(-0.5, 0.5, N)
        EPS = []
        for (lay, d) in zip(self.layers, self.d):
            # EPSt=sum([sub.fou(lay.G[i][0],lay.G[i][1],lay.creator.x_list,lay.creator.y_list,lay.creator.eps_lists)*np.exp((0+2j)*np.pi*(lay.G[i][0]*X+lay.G[i][1]*y)) for i in range(lay.D)])
            EPSt = sum(
                [
                    lay.FOUP[i, lay.D // 2]
                    * np.exp((0 + 2j) * np.pi * (lay.G[i][0] * X + lay.G[i][1] * y))
                    for i in range(lay.D)
                ]
            )
            for i in range(int(d / dz)):
                EPS.append(EPSt)
        EPS = np.array(EPS)
        fig = plt.figure()
        plt.imshow(
            func(EPS).T,
            origin="lower",
            extent=[0.0, sum(self.d), -0.5, 0.5],
            cmap=plt.get_cmap(cmap),
        )
        plt.colorbar()
        sub.savefig(pdf, fig)

    def plot_stack_y(
        self, pdf=None, N=100, dz=0.01, x=0.0, func=np.abs, cmap="viridis"
    ):
        """Plots the stack yz cross section


        Args:
            pdf (multiple, optional): Multiple choice. Each one has a diffrent meaning:
                - (PdfPages): append figure to the PdfPages object
                - (str): save the figure to a pdf with this name
                - None (Default): Do not plot anything. Keep the figure as the active one.
            N (int, optional): Number of points in the x direction. Defaults to 100.
            dz (float, optional): Resolution in z direction (unit of ax). Defaults to 0.01.
            x (float, optional): x coordinate at which the cross section is taken (unit of ax). Defaults to 0.0.
            func (callable, optional): Function to be applied to eps befor plotting. Defaults to np.abs.
            cmap (str, optional): matplotlib colormap for plotting. Defaults to 'viridis'.

        Returns:
            None.

        """
        Y = np.linspace(-0.5, 0.5, N)
        EPS = []
        for (lay, d) in zip(self.layers, self.d):
            EPSt = sum(
                [
                    sub.fou(
                        lay.G[i][0],
                        lay.G[i][1],
                        lay.creator.x_list,
                        lay.creator.y_list,
                        lay.creator.eps_lists,
                    )
                    * np.exp((0 + 2j) * np.pi * (lay.G[i][0] * x + lay.G[i][1] * Y))
                    for i in range(lay.D)
                ]
            )
            for i in range(int(d / dz)):
                EPS.append(EPSt)
        EPS = np.array(EPS)
        fig = plt.figure()
        plt.imshow(
            func(EPS).T,
            origin="lower",
            extent=[0.0, sum(self.d), -0.5, 0.5],
            cmap=plt.get_cmap(cmap),
        )
        plt.colorbar()
        sub.savefig(pdf, fig)

    def count_interface(self):
        """Helper function to identify the different layers and the needed interfaces

        Returns:
            None.

        """
        self.tot_thick = sum(self.d)
        self.lay_list = []
        for lay in self.layers:
            if not lay in self.lay_list:
                self.lay_list.append(lay)
        self.int_list = []
        self.interfaces = []
        for i in range(self.N - 1):
            T_inter = (self.layers[i], self.layers[i + 1])
            if not T_inter in self.int_list:
                self.int_list.append(T_inter)
            self.interfaces.append(T_inter)

    # =============================================================================
    #     def fourier(self,threads=1):
    #         p=Pool(threads)
    #         mat_list=p.map(Layer_empty_st.fourier,self.lay_list)
    #         for lay,FOUP,INV,EPS1,EPS2 in zip(self.lay_list,mat_list):
    #             lay.FOUP=FOUP
    #             lay.INV=INV
    #             lay.EPS1=EPS1
    #             lay.EPS2=EPS2
    #         del mat_list
    #
    # =============================================================================

    def solve(self, k0, kx=0.0, ky=0.0):
        """Calculates the scattering matrix of the multilayer (cpu friendly version)

        This version of solve solve the system in the "smart" way, solving fisrt the eigenvalue problem in each unique layer and the interface matrices of all the interface involved. The computaitonal time scales with the number of different layers, not with the total one.
        It prioritize minimize the calculation done while using more memory.

        Args:
            k0 (float): Vacuum wavevector for the simulation (freqency).
            kx (float, optional): Wavevector in the x direction for the pseudo-fourier expansion. Defaults to 0.0.
            ky (float, optional): Wavevector in the x direction for the pseudo-fourier expansion. Defaults to 0.0.

        Returns:
            None.

        """
        for lay in self.lay_list:
            lay.mode(k0, kx=kx, ky=ky)
            # lay.get_P_norm()
        self.layers[0].get_P_norm()
        self.layers[-1].get_P_norm()
        self.int_matrices = []
        for i in self.int_list:
            self.int_matrices.append(i[0].interface(i[1]))
        self.S = copy.deepcopy(self.int_matrices[0])
        for i in range(1, self.N - 1):
            self.S.add_uniform(self.layers[i], self.d[i])
            self.S.add(self.int_matrices[self.int_list.index(self.interfaces[i])])

    def solve_serial(self, k0, kx=0.0, ky=0.0):
        """Calculates the scattering matrix of the multilayer (memory friendly version)

        This version solves sequentially the layers and the interface as they are in the stack. It is more momery efficient since onlt the data of 2 layer are kept in memory at any given time. Computational time scales with the total number of layer, regardless if they are equal or not.
        It prioritize memory efficiency while possibly requiring more calculations.

        Args:
            k0 (float): Vacuum wavevector for the simulation (freqency).
            kx (float, optional): Wavevector in the x direction for the pseudo-fourier expansion. Defaults to 0.0.
            ky (float, optional): Wavevector in the x direction for the pseudo-fourier expansion. Defaults to 0.0.

        Returns:
            None.

        """
        lay1 = self.layers[0]
        lay1.mode(k0, kx=kx, ky=ky)
        lay1.get_P_norm()
        self.S = S_matrix(2 * self.NPW)
        for i in range(1, self.N):
            lay2 = self.layers[i]
            lay2.mode(k0, kx=kx, ky=ky)
            self.S.add(lay1.interface(lay2))
            self.S.add_uniform(lay2, self.d[i])
            if lay1 is not lay2 and i != 1 and i != self.N:
                lay1.clear()
            lay1 = lay2
        lay2.mode(k0, kx=kx, ky=ky)
        lay2.get_P_norm()

    def solve_lay(self, k0, kx=0.0, ky=0.0):
        """Solve the eigenvalue problem of all the layer in the stack


        Args:
            k0 (float): Vacuum wavevector for the simulation (freqency).
            kx (float, optional): Wavevector in the x direction for the pseudo-fourier expansion. Defaults to 0.0.
            ky (float, optional): Wavevector in the x direction for the pseudo-fourier expansion. Defaults to 0.0.

        Returns:
            None.

        """
        for lay in self.lay_list:
            lay.mode(k0, kx=kx, ky=ky)
            # lay.get_P_norm()
        self.layers[0].get_P_norm()
        self.layers[-1].get_P_norm()

    def solve_S(self):
        """Builds the scattering matrix of the stacks. It assumes that all the layers are alredy solved.

        Returns:
            None.

        """
        self.int_matrices = []
        for i in self.int_list:
            self.int_matrices.append(i[0].interface(i[1]))
        self.S = copy.deepcopy(self.int_matrices[0])
        for i in range(1, self.N - 1):
            self.S.add_uniform(self.layers[i], self.d[i])
            self.S.add(self.int_matrices[self.int_list.index(self.interfaces[i])])

    def get_prop(self, u, list_lay, d=None):
        """Calculates the total poyinting vector in the requiested layers


        Args:
            u (ndarray): array containing the modal coefficient incoming in the first layer.
            list_lay (list of int): indexes of the layer of which to calculate the Poynting vector.
            d (ndarray, optional): array containing the modal coefficient incoming in the last layer. Defaults to None.

        Returns:
            dic (dict): Dictionary of the Poyting vectors in the the layers {layer_index : Poyting vector}

        """
        dic = {}
        u1, d2 = np.zeros((2 * self.NPW), complex), np.zeros((2 * self.NPW), complex)
        u1 = u
        if d != None:
            d2 = d
        (u2, d1) = self.S.output(u1, d2)
        lay = self.layers[0]
        d = self.d[0]
        if 0 in list_lay:
            P = lay.get_Poynting(u1, d1)
            dic[0] = P
        # intermediate layers
        S1 = copy.deepcopy(self.int_matrices[self.int_list.index(self.interfaces[0])])
        for i in range(1, self.N - 1):
            S2 = S_matrix(S1.N)
            for l in range(i, self.N - 1):
                S2.add_uniform(self.layers[l], self.d[l])
                S2.add(self.int_matrices[self.int_list.index(self.interfaces[l])])
            if i in list_lay:
                (ul, dl) = S1.int_f_tot(S2, u1, d2)
                P = self.layers[i].get_Poynting(ul, dl)
                dic[i] = P
            S1.add_uniform(self.layers[i], self.d[i])
            S1.add(self.int_matrices[self.int_list.index(self.interfaces[i])])
        lay = self.layers[-1]
        d = self.d[-1]
        if self.N - 1 in list_lay:
            P = lay.get_Poynting(u2, d2)
            dic[self.N - 1] = P
        return dic

    def get_energybalance(self, u, d=None):
        """Get total energy balance of the stack given the inputs

        Return total power reflected, transmitted and absorbed, normalized to the incidenc power.

        Args:
            u (1darray): Modal coefficient of the left input.
            d (1darray, optional): Modal coefficent of the right input. Defaults to None.

        Returns:
            tuple: tuple contining tree floats with meaning:
                - Total power out from left side (reflection if only u).
                - Total power out from right side (transmission if only u).
                - Total power absorbed in the stack.

        """
        u1, d2, e = (
            np.zeros((2 * self.NPW), complex),
            np.zeros((2 * self.NPW), complex),
            np.zeros((2 * self.NPW), complex),
        )
        u1 = u
        PN = self.layers[0].get_Poynting(u1, e)
        if d is not None:
            d2 = d
            PN -= self.layers[-1].get_Poynting(e, d2)
        (u2, d1) = self.S.output(u1, d2)
        P1 = self.layers[0].get_Poynting(u1, d1)
        P2 = self.layers[-1].get_Poynting(u2, d2)
        return P1 / PN, P2 / PN, (P1 - P2) / PN

    def get_inout(self, u, d=None):
        """Return data about the output of the structure given the input


        Args:
            u (1darray): Vector of the modal coefficents of the right inputs.
            d (1darray, optional): Vector of the modal coefficents of the right inputs. Defaults to None.

        Returns:
            dict: Dictionary containing data of the output:
                - 'left' : (u,d,P): forward modal coefficient, backward modal coefficient and Poyinting vector at the left side.
                - 'right' : (u,d,P): forward modal coefficient, backward modal coefficient and Poyinting vector at the right side.
        """

        u1, d2 = np.zeros((2 * self.NPW), complex), np.zeros((2 * self.NPW), complex)
        u1 = u
        if d != None:
            d2 = d
        (u2, d1) = self.S.output(u1, d2)
        dic = {}
        P = self.layers[0].get_Poynting(u1, d1)
        dic["left"] = (u1, d1, P)
        P = self.layers[-1].get_Poynting(u2, d2)
        dic["right"] = (u2, d2, P)
        return dic

    def get_R(self, i, j, ordered=True):
        """Get relfection coefficient between modes

        Args:
            i (int): Index of the source mode.
            j (int): Index of the target mode.
            ordered (bool, optional): If True, modes are ordered for decrasing effective index, otherwise the order is whatever is returned by the diagonalization routine. Defaults to True.

        Returns:
            float: Reflection between the modes

        """
        if ordered:
            j1 = np.argsort(self.layers[0].W)[-i - 1]
            j2 = np.argsort(self.layers[0].W)[-j - 1]
        else:
            j1 = i
            j2 = j
        return (
            np.abs(self.S.S21[j1, j2]) ** 2
            * self.layers[0].P_norm[j2]
            / self.layers[0].P_norm[j1]
        )

    def get_T(self, i, j, ordered=True):
        """Get transmission coefficient between modes.

        Args:
            i (int): Index of the source mode.
            j (int): Index of the target mode.
            ordered (bool, optional): If True, modes are ordered for decrasing effective index, otherwise the order is whatever is returned by the diagonalization routine. Defaults to True.

        Returns:
            float: Transmission between the modes.

        """
        if ordered:
            j1 = np.argsort(self.layers[0].W)[-i - 1]
            j2 = np.argsort(self.layers[-1].W)[-j - 1]
        else:
            j1 = i
            j2 = j
        return (
            np.abs(self.S.S11[j2, j1]) ** 2
            * self.layers[-1].P_norm[j2]
            / self.layers[0].P_norm[j1]
        )

    def get_PR(self, i, j, ordered=True):
        """Get phase of the relfection coefficient between modes

        Args:
            i (int): Index of the source mode.
            j (int): Index of the target mode.
            ordered (bool, optional): If True, modes are ordered for decrasing effective index, otherwise the order is whatever is returned by the diagonalization routine. Defaults to True.

        Returns:
            float: Phase of reflection between the modes

        """
        if ordered:
            j1 = np.argsort(self.layers[0].W)[-i - 1]
            j2 = np.argsort(self.layers[0].W)[-j - 1]
        else:
            j1 = i
            j2 = j
        return np.angle(self.S.S21[j2, j1])

    def get_PT(self, i, j, ordered=True):
        """Get phase of the transmission coefficient between modes

        Args:
            i (int): Index of the source mode.
            j (int): Index of the target mode.
            ordered (bool, optional): If True, modes are ordered for decrasing effective index, otherwise the order is whatever is returned by the diagonalization routine. Defaults to True.

        Returns:
            float: Phase of transmission between the modes

        """
        if ordered:
            j1 = np.argsort(self.layers[0].W)[-i - 1]
            j2 = np.argsort(self.layers[-1].W)[-j - 1]
        else:
            j1 = i
            j2 = j
        return np.angle(self.S.S11[j2, j1])

    def get_el(self, sel, i, j):
        """Returns element of the scattering matrix

        Note: Modes are ordered for decrasing effective index

        Args:
            sel (str): First index of the matrix.
            i (int): Second index of the matrix.
            j (int): Select the relevand submatrix. Choises are '11', '12', '21', '22'.

        Raises:
            ValueError: If sel in not in the allowed.

        Returns:
            complex: Element of the scattering matrix.

        """
        io = np.argsort(self.layers[0].W)[-i]
        jo = np.argsort(self.layers[-1].W)[-j]
        if sel == "11":
            return self.S.S11[io, jo]
        elif sel == "12":
            return self.S.S12[io, jo]
        elif sel == "21":
            return self.S.S21[io, jo]
        elif sel == "22":
            return self.S.S22[io, jo]
        else:
            raise ValueError(f"Sel {sel} not allowed. Only '11', '12', '21', '22'")

    def mode_T(self, uin, uout):
        d1 = np.zeros((2 * self.NPW), complex)
        [u1, d1] = self.S.output(uin, d1)
        return np.abs(np.dot(np.conj(u1), uout)) ** 2

    def double(self):
        """Compose the scattering matrix of the stack with itself, doubling the structure

        When doing this, the lenght of the first al last layer are ignored (set to 0).
        To function properly hoever they need to be equal (but do not need to have physical meaning)

        Raises:
            RuntimeError: Raised if the stack is not solved yet.

        Returns:
            None.

        """
        try:
            self.S.add(self.S)
        except AttributeError:
            raise RuntimeError("structure not solved yet")

    def join(self, st2):
        """Join the scattering matrix of the structure with the one of a second structure

        When doing this, the lenght of the first al last layeror each stack are ignored (set to 0).
        To function last layer of self and first of st2 need to be equal (but do not need to have physical meaning).
        The condiction used to previoselt solve the stack needs to be the same. This is not checked by the code, so be careful.


        Args:
            st2 (Stack): Stack to which to join self.

        Raises:
            RuntimeError: Raised is one the structure is not solved yet.

        Returns:
            None.

        """
        try:
            self.S
        except AttributeError:
            raise RuntimeError("structure 1 not solved yet")
        try:
            st2.S
        except AttributeError:
            raise RuntimeError("structure 2 not solved yet")
        self.S.add(st2.S)
        l1 = self.layers[:-1]
        l2 = st2.layers[1:]
        self.layers = l1 + l2

    def flip(self):
        """Flip a solved stack

        Flip the stack, swapping the left and right side

        Raises:
            RuntimeError: Raised if the structure is not solved yet.

        Returns:
            None.

        """
        try:
            S = copy.deepcopy(self.S)
            self.S.S11 = S.S22
            self.S.S22 = S.S11
            self.S.S12 = S.S21
            self.S.S21 = S.S12
        except AttributeError:
            raise RuntimeError("structure not solved yet")
        self.layers = self.layers[::-1]
        self.d = self.d[::-1]

    def bloch_modes(self):
        """Calculates Bloch modes of the stack.

        This function assumens the stack to represent the unit cell of a periodic structure, and calculates the corresponding Bloch modes.
        The thickness of the first and last layer are ignored (assumed 0). To work correctly first and last layer needs to be the same.

        Returns:
            TYPE: DESCRIPTION.

        """
        [self.BW, self.BV] = self.S.S_modes()
        self.Bk = -(0.0 + 1j) * np.log(self.BW) / (2.0 * np.pi * self.tot_thick)
        # reorder modes
        ind = np.argsort((0.0 + 1.0j) * self.Bk)
        self.BW = self.BW[ind]
        self.Bk = self.Bk[ind]
        self.BV[:, :] = self.BV[:, ind]
        return self.Bk

    def plot_Ey(self, i, dz=0.01, pdf=None, N=100, y=0.0, func=np.real, s=1):
        u1, d2 = np.zeros((2 * self.NPW), complex), np.zeros((2 * self.NPW), complex)
        u1[np.argsort(self.layers[0].W)[-i]] = 1.0 + 0.0j
        (u2, d1) = self.S.output(u1, d2)
        x = np.linspace(-s * 0.5, s * 0.5, s * N)
        ind = range(2 * self.NPW)
        [X, I] = np.meshgrid(x, ind)
        Em = np.zeros(np.shape(X), complex)
        E = []
        # first layer
        lay = self.layers[0]
        d = self.d[0]
        Em = np.zeros(np.shape(X), complex)
        for j in range(self.NPW):
            Em = np.add(
                Em,
                lay.V[j + self.NPW, I]
                * np.exp(
                    (0.0 + 2.0j)
                    * np.pi
                    * ((lay.G[j][0] + lay.kx) * X + (lay.G[j][1] + lay.ky) * y)
                ),
            )
        for z in np.arange(-d, 0.0, dz):
            Emx = np.add(
                u1 * np.exp((0.0 + 2.0j) * np.pi * lay.k0 * lay.gamma * z),
                d1 * np.exp(-(0.0 + 2.0j) * np.pi * lay.k0 * lay.gamma * z),
            )
            Ez = np.dot(Emx, Em)
            E.append(Ez)
        # intermediate layers
        S1 = copy.deepcopy(self.int_matrices[self.int_list.index(self.interfaces[0])])
        for i in range(1, self.N - 1):
            S2 = S_matrix(S1.N)
            for l in range(i, self.N - 1):
                S2.add_uniform(self.layers[l], self.d[l])
                S2.add(self.int_matrices[self.int_list.index(self.interfaces[l])])
            (ul, dl) = S1.int_f(S2, u1)
            Em = np.zeros(np.shape(X), complex)
            for j in range(self.NPW):
                Em = np.add(
                    Em,
                    self.layers[i].V[j + self.NPW, I]
                    * np.exp(
                        (0.0 + 2.0j)
                        * np.pi
                        * (
                            (self.layers[i].G[j][0] + self.layers[i].kx) * X
                            + (self.layers[i].G[j][1] + self.layers[i].ky) * y
                        )
                    ),
                )
            for z in np.arange(0.0, self.d[i], dz):
                Emx = np.add(
                    ul
                    * np.exp(
                        (0.0 + 2.0j)
                        * np.pi
                        * self.layers[i].k0
                        * self.layers[i].gamma
                        * z
                    ),
                    dl
                    * np.exp(
                        -(0.0 + 2.0j)
                        * np.pi
                        * self.layers[i].k0
                        * self.layers[i].gamma
                        * z
                        - self.d[i]
                    ),
                )
                Ez = np.dot(Emx, Em)
                E.append(Ez)
            S1.add_uniform(self.layers[i], self.d[i])
            S1.add(self.int_matrices[self.int_list.index(self.interfaces[i])])
        # last layer
        lay = self.layers[-1]
        d = self.d[-1]
        Em = np.zeros(np.shape(X), complex)
        for j in range(self.NPW):
            Em = np.add(
                Em,
                lay.V[j + self.NPW, I]
                * np.exp(
                    (0.0 + 2.0j)
                    * np.pi
                    * ((lay.G[j][0] + lay.kx) * X + (lay.G[j][1] + lay.ky) * y)
                ),
            )
        for z in np.arange(0.0, d, dz):
            Emx = np.add(
                u2 * np.exp((0.0 + 2.0j) * np.pi * lay.k0 * lay.gamma * z),
                d2 * np.exp(-(0.0 + 2.0j) * np.pi * lay.k0 * lay.gamma * z),
            )
            Ez = np.dot(Emx, Em)
            E.append(Ez)

        if pdf == None:
            out = PdfPages("Ey.pdf")
        else:
            out = pdf
        plt.figure()
        plt.imshow(
            func(E).T, origin="lower", extent=[0.0, sum(self.d), -0.5, 0.5], cmap="jet"
        )
        # plt.colorbar()
        # plt.savefig('field.png',dpi=900)
        out.savefig(dpi=900)
        plt.close()
        if pdf == None:
            out.close()
        return None

    def plot_Ex(self, i, dz=0.01, pdf=None, N=100, y=0.0, func=np.real, s=1):
        u1, d2 = np.zeros((2 * self.NPW), complex), np.zeros((2 * self.NPW), complex)
        u1[np.argsort(self.layers[0].W)[-i]] = 1.0 + 0.0j
        (u2, d1) = self.S.output(u1, d2)
        x = np.linspace(-s * 0.5, s * 0.5, s * N)
        ind = range(2 * self.NPW)
        [X, I] = np.meshgrid(x, ind)
        Em = np.zeros(np.shape(X), complex)
        E = []
        # first layer
        lay = self.layers[0]
        d = self.d[0]
        Em = np.zeros(np.shape(X), complex)
        for j in range(self.NPW):
            Em = np.add(
                Em,
                lay.V[j, I]
                * np.exp(
                    (0.0 + 2.0j)
                    * np.pi
                    * ((lay.G[j][0] + lay.kx) * X + (lay.G[j][1] + lay.ky) * y)
                ),
            )
        for z in np.arange(-d, 0.0, dz):
            Emx = np.add(
                u1 * np.exp((0.0 + 2.0j) * np.pi * lay.k0 * lay.gamma * z),
                d1 * np.exp(-(0.0 + 2.0j) * np.pi * lay.k0 * lay.gamma * z),
            )
            Ez = np.dot(Emx, Em)
            E.append(Ez)
        # intermediate layers
        S1 = copy.deepcopy(self.int_matrices[self.int_list.index(self.interfaces[0])])
        for i in range(1, self.N - 1):
            S2 = S_matrix(S1.N)
            for l in range(i, self.N - 1):
                S2.add_uniform(self.layers[l], self.d[l])
                S2.add(self.int_matrices[self.int_list.index(self.interfaces[l])])
            (ul, dl) = S1.int_f(S2, u1)
            Em = np.zeros(np.shape(X), complex)
            for j in range(self.NPW):
                Em = np.add(
                    Em,
                    self.layers[i].V[j, I]
                    * np.exp(
                        (0.0 + 2.0j)
                        * np.pi
                        * (
                            (self.layers[i].G[j][0] + self.layers[i].kx) * X
                            + (self.layers[i].G[j][1] + self.layers[i].ky) * y
                        )
                    ),
                )
            for z in np.arange(0.0, self.d[i], dz):
                Emx = np.add(
                    ul
                    * np.exp(
                        (0.0 + 2.0j)
                        * np.pi
                        * self.layers[i].k0
                        * self.layers[i].gamma
                        * z
                    ),
                    dl
                    * np.exp(
                        -(0.0 + 2.0j)
                        * np.pi
                        * self.layers[i].k0
                        * self.layers[i].gamma
                        * z
                        - self.d[i]
                    ),
                )
                Ez = np.dot(Emx, Em)
                E.append(Ez)
            S1.add_uniform(self.layers[i], self.d[i])
            S1.add(self.int_matrices[self.int_list.index(self.interfaces[i])])
        # last layer
        lay = self.layers[-1]
        d = self.d[-1]
        Em = np.zeros(np.shape(X), complex)
        for j in range(self.NPW):
            Em = np.add(
                Em,
                lay.V[j, I]
                * np.exp(
                    (0.0 + 2.0j)
                    * np.pi
                    * ((lay.G[j][0] + lay.kx) * X + (lay.G[j][1] + lay.ky) * y)
                ),
            )
        for z in np.arange(0.0, d, dz):
            Emx = np.add(
                u2 * np.exp((0.0 + 2.0j) * np.pi * lay.k0 * lay.gamma * z),
                d2 * np.exp(-(0.0 + 2.0j) * np.pi * lay.k0 * lay.gamma * z),
            )
            Ez = np.dot(Emx, Em)
            E.append(Ez)

        if pdf == None:
            out = PdfPages("Ex.pdf")
        else:
            out = pdf
        plt.figure()
        plt.imshow(
            func(E).T, origin="lower", extent=[0.0, sum(self.d), -0.5, 0.5], cmap="jet"
        )
        # plt.colorbar()
        # plt.savefig('field.png',dpi=900)
        out.savefig(dpi=900)
        plt.close()
        if pdf == None:
            out.close()
        return None

    def plot_E(
        self,
        i=0,
        dz=0.01,
        pdf=None,
        N=100,
        y=0.0,
        func=np.real,
        s=1,
        ordered="yes",
        title=None,
        cmap="viridis",
    ):
        u1, d2 = np.zeros((2 * self.NPW), complex), np.zeros((2 * self.NPW), complex)
        if ordered == "yes":
            u1[np.argsort(self.layers[0].W)[-i - 1]] = 1.0 + 0.0j
        else:
            u1[i] = 1.0 + 0.0j
        (u2, d1) = self.S.output(u1, d2)
        x = np.linspace(-s * 0.5, s * 0.5, s * N)
        ind = range(2 * self.NPW)
        [X, I] = np.meshgrid(x, ind)
        Ex, Ey = [], []
        # first layer
        lay = self.layers[0]
        d = self.d[0]
        Emx, Emy = np.zeros(np.shape(X), complex), np.zeros(np.shape(X), complex)
        for j in range(self.NPW):
            Emx = np.add(
                Emx,
                lay.V[j, I]
                * np.exp(
                    (0.0 + 2.0j)
                    * np.pi
                    * ((lay.G[j][0] + lay.kx) * X + (lay.G[j][1] + lay.ky) * y)
                ),
            )
            Emy = np.add(
                Emy,
                lay.V[j + self.NPW, I]
                * np.exp(
                    (0.0 + 2.0j)
                    * np.pi
                    * ((lay.G[j][0] + lay.kx) * X + (lay.G[j][1] + lay.ky) * y)
                ),
            )
        for z in np.arange(-d, 0.0, dz):
            Em = np.add(
                u1 * np.exp((0.0 + 2.0j) * np.pi * lay.k0 * lay.gamma * z),
                d1 * np.exp(-(0.0 + 2.0j) * np.pi * lay.k0 * lay.gamma * z),
            )
            Ex.append(np.dot(Em, Emx))
            Ey.append(np.dot(Em, Emy))
        # intermediate layers
        S1 = copy.deepcopy(self.int_matrices[self.int_list.index(self.interfaces[0])])
        for i in range(1, self.N - 1):
            S2 = S_matrix(S1.N)
            for l in range(i, self.N - 1):
                S2.add_uniform(self.layers[l], self.d[l])
                S2.add(self.int_matrices[self.int_list.index(self.interfaces[l])])
            (ul, dl) = S1.int_f(S2, u1)
            Emx, Emy = np.zeros(np.shape(X), complex), np.zeros(np.shape(X), complex)
            for j in range(self.NPW):
                Emx = np.add(
                    Emx,
                    self.layers[i].V[j, I]
                    * np.exp(
                        (0.0 + 2.0j)
                        * np.pi
                        * (
                            (self.layers[i].G[j][0] + self.layers[i].kx) * X
                            + (self.layers[i].G[j][1] + self.layers[i].ky) * y
                        )
                    ),
                )
                Emy = np.add(
                    Emy,
                    self.layers[i].V[j + self.NPW, I]
                    * np.exp(
                        (0.0 + 2.0j)
                        * np.pi
                        * (
                            (self.layers[i].G[j][0] + self.layers[i].kx) * X
                            + (self.layers[i].G[j][1] + self.layers[i].ky) * y
                        )
                    ),
                )
            for z in np.arange(0.0, self.d[i], dz):
                Em = np.add(
                    ul
                    * np.exp(
                        (0.0 + 2.0j)
                        * np.pi
                        * self.layers[i].k0
                        * self.layers[i].gamma
                        * z
                    ),
                    dl
                    * np.exp(
                        -(0.0 + 2.0j)
                        * np.pi
                        * self.layers[i].k0
                        * self.layers[i].gamma
                        * z
                    ),
                )
                Ex.append(np.dot(Em, Emx))
                Ey.append(np.dot(Em, Emy))
            S1.add_uniform(self.layers[i], self.d[i])
            S1.add(self.int_matrices[self.int_list.index(self.interfaces[i])])
        # last layer
        lay = self.layers[-1]
        d = self.d[-1]
        Emx, Emy = np.zeros(np.shape(X), complex), np.zeros(np.shape(X), complex)
        for j in range(self.NPW):
            Emx = np.add(
                Emx,
                lay.V[j, I]
                * np.exp(
                    (0.0 + 2.0j)
                    * np.pi
                    * ((lay.G[j][0] + lay.kx) * X + (lay.G[j][1] + lay.ky) * y)
                ),
            )
            Emy = np.add(
                Emy,
                lay.V[j + self.NPW, I]
                * np.exp(
                    (0.0 + 2.0j)
                    * np.pi
                    * ((lay.G[j][0] + lay.kx) * X + (lay.G[j][1] + lay.ky) * y)
                ),
            )
        for z in np.arange(0.0, d, dz):
            Em = np.add(
                u2 * np.exp((0.0 + 2.0j) * np.pi * lay.k0 * lay.gamma * z),
                d2 * np.exp(-(0.0 + 2.0j) * np.pi * lay.k0 * lay.gamma * z),
            )
            Ex.append(np.dot(Em, Emx))
            Ey.append(np.dot(Em, Emy))
        if pdf is not None:
            if isinstance(pdf, PdfPages):
                out = pdf
            else:
                out = PdfPages(pdf)
        plt.figure()
        if title != None:
            plt.suptitle(title)
        plt.subplot(211)
        plt.title("Ex")
        plt.imshow(
            func(Ex).T,
            origin="lower",
            extent=[0.0, sum(self.d), -0.5, 0.5],
            cmap=plt.get_cmap(cmap),
        )
        plt.colorbar()
        plt.subplot(212)
        plt.title("Ey")
        plt.imshow(
            func(Ey).T,
            origin="lower",
            extent=[0.0, sum(self.d), -0.5, 0.5],
            cmap=plt.get_cmap(cmap),
        )
        plt.colorbar()
        # plt.savefig('field.png',dpi=900)
        if pdf is not None:
            out.savefig()
            plt.close()
            if isinstance(pdf, str):
                out.close()
        return None

    def writeE(
        self,
        i=1,
        filename="field.out",
        dz=0.01,
        N=100,
        y=0.0,
        func=np.real,
        s=1,
        ordered="yes",
    ):
        f = open(filename, "w")
        u1, d2 = np.zeros((2 * self.NPW), complex), np.zeros((2 * self.NPW), complex)
        if ordered == "yes":
            u1[np.argsort(self.layers[0].W)[-i]] = 1.0 + 0.0j
        else:
            u1[i] = 1.0 + 0.0j
        (u2, d1) = self.S.output(u1, d2)
        x = np.linspace(-s * 0.5, s * 0.5, s * N)
        ind = range(2 * self.NPW)
        [X, I] = np.meshgrid(x, ind)
        Ex, Ey = [], []
        # first layer
        lay = self.layers[0]
        d = self.d[0]
        Emx, Emy = np.zeros(np.shape(X), complex), np.zeros(np.shape(X), complex)
        zz = 0.0
        for j in range(self.NPW):
            Emx = np.add(
                Emx,
                lay.V[j, I]
                * np.exp(
                    (0.0 + 2.0j)
                    * np.pi
                    * ((lay.G[j][0] + lay.kx) * X + (lay.G[j][1] + lay.ky) * y)
                ),
            )
            Emy = np.add(
                Emy,
                lay.V[j + self.NPW, I]
                * np.exp(
                    (0.0 + 2.0j)
                    * np.pi
                    * ((lay.G[j][0] + lay.kx) * X + (lay.G[j][1] + lay.ky) * y)
                ),
            )
        for z in np.arange(-d, 0.0, dz):
            Em = np.add(
                u1 * np.exp((0.0 + 2.0j) * np.pi * lay.k0 * lay.gamma * z),
                d1 * np.exp(-(0.0 + 2.0j) * np.pi * lay.k0 * lay.gamma * z),
            )
            Ex = np.dot(Em, Emx)
            Ey = np.dot(Em, Emy)
            for i in range(len(x)):
                f.write(
                    8
                    * "%12.6f"
                    % (
                        x[i],
                        zz,
                        Ex[i].real,
                        Ex[i].imag,
                        abs(Ex[i]),
                        Ey[i].real,
                        Ey[i].imag,
                        abs(Ey[i]),
                    )
                    + "\n"
                )
            f.write("\n")
            zz += dz
        # intermediate layers
        S1 = copy.deepcopy(self.int_matrices[self.int_list.index(self.interfaces[0])])
        for i in range(1, self.N - 1):
            S2 = S_matrix(S1.N)
            for l in range(i, self.N - 1):
                S2.add_uniform(self.layers[l], self.d[l])
                S2.add(self.int_matrices[self.int_list.index(self.interfaces[l])])
            (ul, dl) = S1.int_f(S2, u1)
            Emx, Emy = np.zeros(np.shape(X), complex), np.zeros(np.shape(X), complex)
            for j in range(self.NPW):
                Emx = np.add(
                    Emx,
                    self.layers[i].V[j, I]
                    * np.exp(
                        (0.0 + 2.0j)
                        * np.pi
                        * (
                            (self.layers[i].G[j][0] + self.layers[i].kx) * X
                            + (self.layers[i].G[j][1] + self.layers[i].ky) * y
                        )
                    ),
                )
                Emy = np.add(
                    Emy,
                    self.layers[i].V[j + self.NPW, I]
                    * np.exp(
                        (0.0 + 2.0j)
                        * np.pi
                        * (
                            (self.layers[i].G[j][0] + self.layers[i].kx) * X
                            + (self.layers[i].G[j][1] + self.layers[i].ky) * y
                        )
                    ),
                )
            for z in np.arange(0.0, self.d[i], dz):
                Em = np.add(
                    ul
                    * np.exp(
                        (0.0 + 2.0j)
                        * np.pi
                        * self.layers[i].k0
                        * self.layers[i].gamma
                        * z
                    ),
                    dl
                    * np.exp(
                        -(0.0 + 2.0j)
                        * np.pi
                        * self.layers[i].k0
                        * self.layers[i].gamma
                        * z
                    ),
                )
                Ex = np.dot(Em, Emx)
                Ey = np.dot(Em, Emy)
                for ii in range(len(x)):
                    f.write(
                        8
                        * "%12.6f"
                        % (
                            x[ii],
                            zz,
                            Ex[ii].real,
                            Ex[ii].imag,
                            abs(Ex[ii]),
                            Ey[ii].real,
                            Ey[ii].imag,
                            abs(Ey[ii]),
                        )
                        + "\n"
                    )
                f.write("\n")
                zz += dz
            S1.add_uniform(self.layers[i], self.d[i])
            S1.add(self.int_matrices[self.int_list.index(self.interfaces[i])])
        # last layer
        lay = self.layers[-1]
        d = self.d[-1]
        Emx, Emy = np.zeros(np.shape(X), complex), np.zeros(np.shape(X), complex)
        for j in range(self.NPW):
            Emx = np.add(
                Emx,
                lay.V[j, I]
                * np.exp(
                    (0.0 + 2.0j)
                    * np.pi
                    * ((lay.G[j][0] + lay.kx) * X + (lay.G[j][1] + lay.ky) * y)
                ),
            )
            Emy = np.add(
                Emy,
                lay.V[j + self.NPW, I]
                * np.exp(
                    (0.0 + 2.0j)
                    * np.pi
                    * ((lay.G[j][0] + lay.kx) * X + (lay.G[j][1] + lay.ky) * y)
                ),
            )
        for z in np.arange(0.0, d, dz):
            Em = np.add(
                u2 * np.exp((0.0 + 2.0j) * np.pi * lay.k0 * lay.gamma * z),
                d2 * np.exp(-(0.0 + 2.0j) * np.pi * lay.k0 * lay.gamma * z),
            )
            for i in range(len(x)):
                f.write(
                    8
                    * "%12.6f"
                    % (
                        x[i],
                        zz,
                        Ex[i].real,
                        Ex[i].imag,
                        abs(Ex[i]),
                        Ey[i].real,
                        Ey[i].imag,
                        abs(Ey[i]),
                    )
                    + "\n"
                )
            f.write("\n")
            zz += dz
        f.close()

    def plot_E_plane(
        self,
        i,
        jlay,
        z,
        N=100,
        pdf=None,
        pdfname=None,
        func=np.real,
        s=1,
        ordered="yes",
        title=None,
    ):
        u1, d2 = np.zeros((2 * self.NPW), complex), np.zeros((2 * self.NPW), complex)
        if ordered == "yes":
            u1[np.argsort(self.layers[0].W)[-i]] = 1.0 + 0.0j
        else:
            u1[i] = 1.0 + 0.0j
        (u2, d1) = self.S.output(u1, d2)
        [X, Y] = np.meshgrid(
            np.linspace(-s * 0.5, s * 0.5, s * N),
            np.linspace(
                -s * 0.5 * self.layers[jlay].Nyx, s * 0.5 * self.layers[jlay].Nyx, s * N
            ),
        )
        S1 = copy.deepcopy(self.int_matrices[self.int_list.index(self.interfaces[0])])
        S2 = S_matrix(S1.N)
        for l in range(1, jlay):
            S1.add_uniform(self.layers[l], self.d[l])
            S1.add(self.int_matrices[self.int_list.index(self.interfaces[l])])
        for l in range(jlay, self.N - 1):
            S2.add_uniform(self.layers[l], self.d[l])
            S2.add(self.int_matrices[self.int_list.index(self.interfaces[l])])
        (ul, dl) = S1.int_f(S2, u1)
        Emx_l, Emy_l = [], []
        for i in range(2 * self.NPW):
            Emx, Emy = np.zeros(np.shape(X), complex), np.zeros(np.shape(X), complex)
            for j in range(self.NPW):
                Emx = np.add(
                    Emx,
                    self.layers[jlay].V[j, i]
                    * np.exp(
                        (0.0 + 2.0j)
                        * np.pi
                        * (
                            (self.layers[jlay].G[j][0] + self.layers[jlay].kx) * X
                            + (self.layers[jlay].G[j][1] + self.layers[jlay].ky) * Y
                        )
                    ),
                )
                Emy = np.add(
                    Emy,
                    self.layers[jlay].V[j + self.NPW, i]
                    * np.exp(
                        (0.0 + 2.0j)
                        * np.pi
                        * (
                            (self.layers[jlay].G[j][0] + self.layers[jlay].kx) * X
                            + (self.layers[jlay].G[j][1] + self.layers[jlay].ky) * Y
                        )
                    ),
                )
            Emx_l.append(Emx)
            Emy_l.append(Emy)
        Em = np.add(
            ul
            * np.exp(
                (0.0 + 2.0j)
                * np.pi
                * self.layers[jlay].k0
                * self.layers[jlay].gamma
                * z
                * self.d[jlay]
            ),
            dl
            * np.exp(
                -(0.0 + 2.0j)
                * np.pi
                * self.layers[jlay].k0
                * self.layers[jlay].gamma
                * z
                * self.d[jlay]
            ),
        )
        Ex, Ey = np.zeros(np.shape(X), complex), np.zeros(np.shape(X), complex)
        for i in range(2 * self.NPW):
            Ex = np.add(Ex, Em[i] * Emx_l[i])
            Ey = np.add(Ey, Em[i] * Emy_l[i])
        if pdf == None:
            if pdfname != None:
                out = PdfPages(pdfname + ".pdf")
            else:
                out = PdfPages("E_plane.pdf")
        else:
            out = pdf
        plt.figure()
        if title != None:
            plt.suptitle(title)
        plt.subplot(211)
        plt.title("Ex")
        plt.imshow(func(Ex), origin="lower", extent=[-0.5, 0.5, -0.5, 0.5], cmap="jet")
        plt.colorbar()
        plt.subplot(212)
        plt.title("Ey")
        plt.imshow(func(Ey), origin="lower", extent=[-0.5, 0.5, -0.5, 0.5], cmap="jet")
        plt.colorbar()
        # plt.savefig('field.png',dpi=900)
        out.savefig(dpi=900)
        plt.close()
        if pdf == None:
            out.close()
        return None

    def plot_EY(
        self,
        i=1,
        dz=0.01,
        pdf=None,
        pdfname=None,
        N=100,
        x=0.0,
        func=np.real,
        s=1,
        ordered="yes",
        title=None,
    ):
        u1, d2 = np.zeros((2 * self.NPW), complex), np.zeros((2 * self.NPW), complex)
        if ordered == "yes":
            u1[np.argsort(self.layers[0].W)[-i]] = 1.0 + 0.0j
        else:
            u1[i] = 1.0 + 0.0j
        (u2, d1) = self.S.output(u1, d2)
        y = np.linspace(-s * 0.5, s * 0.5, s * N)
        ind = range(2 * self.NPW)
        [Y, I] = np.meshgrid(y, ind)
        Ex, Ey = [], []
        # first layer
        lay = self.layers[0]
        d = self.d[0]
        Emx, Emy = np.zeros(np.shape(Y), complex), np.zeros(np.shape(Y), complex)
        for j in range(self.NPW):
            Emx = np.add(
                Emx,
                lay.V[j, I]
                * np.exp(
                    (0.0 + 2.0j)
                    * np.pi
                    * ((lay.G[j][0] + lay.kx) * x + (lay.G[j][1] + lay.ky) * Y)
                ),
            )
            Emy = np.add(
                Emy,
                lay.V[j + self.NPW, I]
                * np.exp(
                    (0.0 + 2.0j)
                    * np.pi
                    * ((lay.G[j][0] + lay.kx) * x + (lay.G[j][1] + lay.ky) * Y)
                ),
            )
        for z in np.arange(-d, 0.0, dz):
            Em = np.add(
                u1 * np.exp((0.0 + 2.0j) * np.pi * lay.k0 * lay.gamma * z),
                d1 * np.exp(-(0.0 + 2.0j) * np.pi * lay.k0 * lay.gamma * z),
            )
            Ex.append(np.dot(Em, Emx))
            Ey.append(np.dot(Em, Emy))
        # intermediate layers
        S1 = copy.deepcopy(self.int_matrices[self.int_list.index(self.interfaces[0])])
        for i in range(1, self.N - 1):
            S2 = S_matrix(S1.N)
            for l in range(i, self.N - 1):
                S2.add_uniform(self.layers[l], self.d[l])
                S2.add(self.int_matrices[self.int_list.index(self.interfaces[l])])
            (ul, dl) = S1.int_f(S2, u1)
            Emx, Emy = np.zeros(np.shape(Y), complex), np.zeros(np.shape(Y), complex)
            for j in range(self.NPW):
                Emx = np.add(
                    Emx,
                    self.layers[i].V[j, I]
                    * np.exp(
                        (0.0 + 2.0j)
                        * np.pi
                        * (
                            (self.layers[i].G[j][0] + self.layers[i].kx) * x
                            + (self.layers[i].G[j][1] + self.layers[i].ky) * Y
                        )
                    ),
                )
                Emy = np.add(
                    Emy,
                    self.layers[i].V[j + self.NPW, I]
                    * np.exp(
                        (0.0 + 2.0j)
                        * np.pi
                        * (
                            (self.layers[i].G[j][0] + self.layers[i].kx) * x
                            + (self.layers[i].G[j][1] + self.layers[i].ky) * Y
                        )
                    ),
                )
            for z in np.arange(0.0, self.d[i], dz):
                Em = np.add(
                    ul
                    * np.exp(
                        (0.0 + 2.0j)
                        * np.pi
                        * self.layers[i].k0
                        * self.layers[i].gamma
                        * z
                    ),
                    dl
                    * np.exp(
                        -(0.0 + 2.0j)
                        * np.pi
                        * self.layers[i].k0
                        * self.layers[i].gamma
                        * z
                    ),
                )
                Ex.append(np.dot(Em, Emx))
                Ey.append(np.dot(Em, Emy))
            S1.add_uniform(self.layers[i], self.d[i])
            S1.add(self.int_matrices[self.int_list.index(self.interfaces[i])])
        # last layer
        lay = self.layers[-1]
        d = self.d[-1]
        Emx, Emy = np.zeros(np.shape(Y), complex), np.zeros(np.shape(Y), complex)
        for j in range(self.NPW):
            Emx = np.add(
                Emx,
                lay.V[j, I]
                * np.exp(
                    (0.0 + 2.0j)
                    * np.pi
                    * ((lay.G[j][0] + lay.kx) * x + (lay.G[j][1] + lay.ky) * Y)
                ),
            )
            Emy = np.add(
                Emy,
                lay.V[j + self.NPW, I]
                * np.exp(
                    (0.0 + 2.0j)
                    * np.pi
                    * ((lay.G[j][0] + lay.kx) * x + (lay.G[j][1] + lay.ky) * Y)
                ),
            )
        for z in np.arange(0.0, d, dz):
            Em = np.add(
                u2 * np.exp((0.0 + 2.0j) * np.pi * lay.k0 * lay.gamma * z),
                d2 * np.exp(-(0.0 + 2.0j) * np.pi * lay.k0 * lay.gamma * z),
            )
            Ex.append(np.dot(Em, Emx))
            Ey.append(np.dot(Em, Emy))
        if pdf == None:
            out = PdfPages("EY.pdf")
        else:
            out = pdf
        plt.figure()
        if title != None:
            plt.suptitle(title)
        plt.subplot(211)
        plt.title("Ex")
        plt.imshow(
            func(Ex).T, origin="lower", extent=[0.0, sum(self.d), -0.5, 0.5], cmap="jet"
        )
        plt.colorbar()
        plt.subplot(212)
        plt.title("Ey")
        plt.imshow(
            func(Ey).T, origin="lower", extent=[0.0, sum(self.d), -0.5, 0.5], cmap="jet"
        )
        plt.colorbar()
        # plt.savefig('field.png',dpi=900)
        out.savefig(dpi=900)
        plt.close()
        if pdf == None:
            out.close()
        return None

    def plot_E_periodic(
        self,
        ii,
        r=1,
        dz=0.01,
        pdf=None,
        N=100,
        y=0.0,
        func=np.real,
        s=1,
        title=None,
        figsize=(12, 6),
    ):
        [u, d] = np.split(self.BV[:, ii], 2)
        d = d * self.BW[ii]
        x = np.linspace(-s * 0.5, s * 0.5, s * N)
        ind = range(2 * self.NPW)
        [X, I] = np.meshgrid(x, ind)
        Ex, Ey = [], []
        # intermediate layers
        S1 = copy.deepcopy(self.int_matrices[self.int_list.index(self.interfaces[0])])
        for i in range(1, self.N - 1):
            S2 = S_matrix(S1.N)
            for l in range(i, self.N - 1):
                S2.add_uniform(self.layers[l], self.d[l])
                S2.add(self.int_matrices[self.int_list.index(self.interfaces[l])])
            (ul, dl) = S1.int_complete(S2, u, d)
            Emx, Emy = np.zeros(np.shape(X), complex), np.zeros(np.shape(X), complex)
            for j in range(self.NPW):
                Emx = np.add(
                    Emx,
                    self.layers[i].V[j, I]
                    * np.exp(
                        (0.0 + 2.0j)
                        * np.pi
                        * (
                            (self.layers[i].G[j][0] + self.layers[i].kx) * X
                            + (self.layers[i].G[j][1] + self.layers[i].ky) * y
                        )
                    ),
                )
                Emy = np.add(
                    Emy,
                    self.layers[i].V[j + self.NPW, I]
                    * np.exp(
                        (0.0 + 2.0j)
                        * np.pi
                        * (
                            (self.layers[i].G[j][0] + self.layers[i].kx) * X
                            + (self.layers[i].G[j][1] + self.layers[i].ky) * y
                        )
                    ),
                )
            for z in np.arange(0.0, self.d[i], dz):
                Em = np.add(
                    ul
                    * np.exp(
                        (0.0 + 2.0j)
                        * np.pi
                        * self.layers[i].k0
                        * self.layers[i].gamma
                        * z
                    ),
                    dl
                    * np.exp(
                        -(0.0 + 2.0j)
                        * np.pi
                        * self.layers[i].k0
                        * self.layers[i].gamma
                        * z
                    ),
                )
                Ex.append(np.dot(Em, Emx))
                Ey.append(np.dot(Em, Emy))
            S1.add_uniform(self.layers[i], self.d[i])
            S1.add(self.int_matrices[self.int_list.index(self.interfaces[i])])
        Ex, Ey = np.array(Ex), np.array(Ey)
        # print ii,np.abs([self.BW[ii]**k for k in range(r)])
        Ex = np.vstack([self.BW[ii] ** k * Ex for k in range(r)])
        Ey = np.vstack([self.BW[ii] ** k * Ey for k in range(r)])
        fig = plt.figure(figsize=figsize)
        plt.subplot(211)
        plt.title("Ex")
        plt.imshow(
            func(Ex).T,
            origin="lower",
            extent=[0.0, r * sum(self.d), -0.5, 0.5],
            cmap="jet",
        )
        plt.colorbar()
        plt.subplot(212)
        plt.title("Ey")
        plt.imshow(
            func(Ey).T,
            origin="lower",
            extent=[0.0, r * sum(self.d), -0.5, 0.5],
            cmap="jet",
        )
        plt.colorbar()
        if title != None:
            plt.suptitle(title)
        sub.savefig(pdf, fig)
        return None

    def writeE_periodic_XZ(
        self, ii, r=1, filename="fieldE_XZ.out", dz=0.01, N=100, y=0.0, s=1.0
    ):
        """Write to a file the filed component of the Bloch mode (XZ cross section) of a periodic structure.


        Args:
            ii (int): Index of the Bloch mode to be plotted.
            r (int, optional): Number unit cells plotted in z direction. Defaults to 1.
            filename (str, optional): Name of the file for writing the field. Defaults to 'fieldE_XZ.out'.
            dz (float, optional): Grid dimension in z. Defaults to 0.01.
            N (int, optional): Number of points in x direction. Defaults to 100.
            y (float, optional): y coordinate of the cross section. Defaults to 0.0.
            s (float, optional): Width of the plot in x (unit of ax). Defaults to 1.0.

        Returns:
            tuple: tutple containing:
                - 1darray: x coordinate in the unit cell
                - 1darray: x coordintate in the real space (only meaniningful if coordinate transformation is applied)
                - 1darray: z coordinate
                - 2darray: Ex
                - 2darray: Ey

        """
        [u, d] = np.split(self.BV[:, ii], 2)
        d = d * self.BW[ii]
        try:
            ex = self.ex
            if s == 1.0:
                x = np.linspace(-0.5, 0.5, N)
                xl = sub.t_dir(x, ex)
            else:
                xl = np.linspace(-s * 0.5, s * 0.5, N)
                x = sub.t_inv(xl, ex)
        except AttributeError:
            x = np.linspace(-0.5, 0.5, N)
            xl = np.linspace(-0.5, 0.5, N)
        ind = range(2 * self.NPW)
        [X, I] = np.meshgrid(x, ind)
        Ex, Ey = [], []
        # intermediate layers
        S1 = copy.deepcopy(self.int_matrices[self.int_list.index(self.interfaces[0])])
        for i in range(1, self.N - 1):
            S2 = S_matrix(S1.N)
            for l in range(i, self.N - 1):
                S2.add_uniform(self.layers[l], self.d[l])
                S2.add(self.int_matrices[self.int_list.index(self.interfaces[l])])
            (ul, dl) = S1.int_complete(S2, u, d)
            Emx, Emy = np.zeros(np.shape(X), complex), np.zeros(np.shape(X), complex)
            for j in range(self.NPW):
                Emx = np.add(
                    Emx,
                    self.layers[i].V[j, I]
                    * np.exp(
                        (0.0 + 2.0j)
                        * np.pi
                        * (
                            (self.layers[i].G[j][0] + self.layers[i].kx) * X
                            + (self.layers[i].G[j][1] + self.layers[i].ky) * y
                        )
                    ),
                )
                Emy = np.add(
                    Emy,
                    self.layers[i].V[j + self.NPW, I]
                    * np.exp(
                        (0.0 + 2.0j)
                        * np.pi
                        * (
                            (self.layers[i].G[j][0] + self.layers[i].kx) * X
                            + (self.layers[i].G[j][1] + self.layers[i].ky) * y
                        )
                    ),
                )
            start = 0.0 if i == 1 else dz
            for z in np.arange(start, self.d[i], dz):
                Em = np.add(
                    ul
                    * np.exp(
                        (0.0 + 2.0j)
                        * np.pi
                        * self.layers[i].k0
                        * self.layers[i].gamma
                        * z
                    ),
                    dl
                    * np.exp(
                        -(0.0 + 2.0j)
                        * np.pi
                        * self.layers[i].k0
                        * self.layers[i].gamma
                        * z
                    ),
                )
                Ex.append(np.dot(Em, Emx))
                Ey.append(np.dot(Em, Emy))
            S1.add_uniform(self.layers[i], self.d[i])
            S1.add(self.int_matrices[self.int_list.index(self.interfaces[i])])
        Ex, Ey = np.array(Ex), np.array(Ey)
        # print ii,np.abs([self.BW[ii]**k for k in range(r)])
        Ex = np.vstack([self.BW[ii] ** k * Ex for k in range(r)])
        Ey = np.vstack([self.BW[ii] ** k * Ey for k in range(r)])
        f = open(filename, "w")
        f.write(
            "#    x           xt         z           ReEx        ImEx        AbsEx       ReEy        ImEy        AbsEy \n"
        )
        for i in range(np.shape(Ex)[0]):
            for j in range(np.shape(Ex)[1]):
                f.write(
                    9
                    * "%15.6e"
                    % (
                        x[j],
                        xl[j],
                        dz * i,
                        Ex[i, j].real,
                        Ex[i, j].imag,
                        abs(Ex[i, j]),
                        Ey[i, j].real,
                        Ey[i, j].imag,
                        abs(Ey[i, j]),
                    )
                    + "\n"
                )
            f.write("\n")
        f.close()
        DZ = np.array([dz * i for i in range(np.shape(Ex)[0])])
        return x, xl, DZ, Ex, Ey

    def writeE_periodic_YZ(
        self, ii, r=1, filename="fieldE_YZ.out", dz=0.01, N=100, x=0.0, s=1.0
    ):
        """Write to a file the filed component of the Bloch mode (YZ cross section) of a periodic structure.


        Args:
            ii (int): Index of the Bloch mode to be plotted.
            r (int, optional): Number unit cells plotted in z direction. Defaults to 1.
            filename (str, optional): Name of the file for writing the field. Defaults to 'fieldE_XZ.out'.
            dz (float, optional): Grid dimension in z. Defaults to 0.01.
            N (int, optional): Number of points in y direction. Defaults to 100.
            x (float, optional): x coordinate of the cross section. Defaults to 0.0.
            s (float, optional): Width of the plot in y (unit of ay). Defaults to 1.0.

        Returns:
            tuple: tutple containing:
                - 1darray: y coordinate in the unit cell
                - 1darray: y coordintate in the real space (only meaniningful if coordinate transformation is applied)
                - 1darray: z coordinate
                - 2darray: Ex
                - 2darray: Ey

        """
        [u, d] = np.split(self.BV[:, ii], 2)
        d = d * self.BW[ii]
        try:
            ey = self.ey
            if s == 1.0:
                y = np.linspace(-0.5, 0.5, N)
                yl = sub.t_dir(y, ey)
            else:
                yl = np.linspace(-s * 0.5, s * 0.5, N)
                y = sub.t_inv(yl, ey)
        except AttributeError:
            y = np.linspace(-0.5, 0.5, N)
            yl = np.linspace(-0.5, 0.5, N)
        ind = range(2 * self.NPW)
        [Y, I] = np.meshgrid(y, ind)
        Ex, Ey = [], []
        # intermediate layers
        S1 = copy.deepcopy(self.int_matrices[self.int_list.index(self.interfaces[0])])
        for i in range(1, self.N - 1):
            S2 = S_matrix(S1.N)
            for l in range(i, self.N - 1):
                S2.add_uniform(self.layers[l], self.d[l])
                S2.add(self.int_matrices[self.int_list.index(self.interfaces[l])])
            (ul, dl) = S1.int_complete(S2, u, d)
            Emx, Emy = np.zeros(np.shape(Y), complex), np.zeros(np.shape(Y), complex)
            for j in range(self.NPW):
                Emx = np.add(
                    Emx,
                    self.layers[i].V[j, I]
                    * np.exp(
                        (0.0 + 2.0j)
                        * np.pi
                        * (
                            (self.layers[i].G[j][0] + self.layers[i].kx) * x
                            + (self.layers[i].G[j][1] + self.layers[i].ky) * Y
                        )
                    ),
                )
                Emy = np.add(
                    Emy,
                    self.layers[i].V[j + self.NPW, I]
                    * np.exp(
                        (0.0 + 2.0j)
                        * np.pi
                        * (
                            (self.layers[i].G[j][0] + self.layers[i].kx) * x
                            + (self.layers[i].G[j][1] + self.layers[i].ky) * Y
                        )
                    ),
                )
            start = 0.0 if i == 1 else dz
            for z in np.arange(start, self.d[i], dz):
                Em = np.add(
                    ul
                    * np.exp(
                        (0.0 + 2.0j)
                        * np.pi
                        * self.layers[i].k0
                        * self.layers[i].gamma
                        * z
                    ),
                    dl
                    * np.exp(
                        -(0.0 + 2.0j)
                        * np.pi
                        * self.layers[i].k0
                        * self.layers[i].gamma
                        * z
                    ),
                )
                Ex.append(np.dot(Em, Emx))
                Ey.append(np.dot(Em, Emy))
            S1.add_uniform(self.layers[i], self.d[i])
            S1.add(self.int_matrices[self.int_list.index(self.interfaces[i])])
        Ex, Ey = np.array(Ex), np.array(Ey)
        # print ii,np.abs([self.BW[ii]**k for k in range(r)])
        Ex = np.vstack([self.BW[ii] ** k * Ex for k in range(r)])
        Ey = np.vstack([self.BW[ii] ** k * Ey for k in range(r)])
        f = open(filename, "w")
        f.write(
            "#    y           yt         z           ReEx        ImEx        AbsEx       ReEy        ImEy        AbsEy \n"
        )
        for i in range(np.shape(Ex)[0]):
            for j in range(np.shape(Ex)[1]):
                f.write(
                    9
                    * "%15.6e"
                    % (
                        y[j] * self.Nyx,
                        yl[j] * self.Nyx,
                        dz * i,
                        Ex[i, j].real,
                        Ex[i, j].imag,
                        abs(Ex[i, j]),
                        Ey[i, j].real,
                        Ey[i, j].imag,
                        abs(Ey[i, j]),
                    )
                    + "\n"
                )
            f.write("\n")
        f.close()
        DZ = np.array([dz * i for i in range(np.shape(Ex)[0])])
        return y, yl, DZ, Ex, Ey

    def writeE_periodic_XY(self, ii, jlay, z, filename="fieldE_XY.out", N=100, s=1.0):
        [u, d] = np.split(self.BV[:, ii], 2)
        d = d * self.BW[ii]
        try:
            ex = self.ex
            if s == 1.0:
                x = np.linspace(-0.5, 0.5, N)
                xl = sub.t_dir(x, ex)
            else:
                xl = np.linspace(-s * 0.5, s * 0.5, N)
                x = sub.t_inv(xl, ex)
        except AttributeError:
            x = np.linspace(-0.5, 0.5, N)
            xl = np.linspace(-0.5, 0.5, N)
        try:
            ey = self.ey
            if s == 1.0:
                y = np.linspace(-0.5, 0.5, N)
                yl = sub.t_dir(y, ey)
            else:
                yl = np.linspace(-s * 0.5, s * 0.5, N)
                y = sub.t_inv(yl, ey)
        except AttributeError:
            y = np.linspace(-0.5, 0.5, N)
            yl = np.linspace(-0.5, 0.5, N)
        ind = range(2 * self.NPW)
        [X, Y] = np.meshgrid(x, y)
        S1 = copy.deepcopy(self.int_matrices[self.int_list.index(self.interfaces[0])])
        S2 = S_matrix(S1.N)
        for l in range(1, jlay):
            S1.add_uniform(self.layers[l], self.d[l])
            S1.add(self.int_matrices[self.int_list.index(self.interfaces[l])])
        for l in range(jlay, self.N - 1):
            S2.add_uniform(self.layers[l], self.d[l])
            S2.add(self.int_matrices[self.int_list.index(self.interfaces[l])])
        (ul, dl) = S1.int_complete(S2, u, d)
        Emx_l, Emy_l = [], []
        for i in range(2 * self.NPW):
            Emx, Emy = np.zeros(np.shape(X), complex), np.zeros(np.shape(X), complex)
            for j in range(self.NPW):
                Emx = np.add(
                    Emx,
                    self.layers[jlay].V[j, i]
                    * np.exp(
                        (0.0 + 2.0j)
                        * np.pi
                        * (
                            (self.layers[jlay].G[j][0] + self.layers[jlay].kx) * X
                            + (self.layers[jlay].G[j][1] + self.layers[jlay].ky) * Y
                        )
                    ),
                )
                Emy = np.add(
                    Emy,
                    self.layers[jlay].V[j + self.NPW, i]
                    * np.exp(
                        (0.0 + 2.0j)
                        * np.pi
                        * (
                            (self.layers[jlay].G[j][0] + self.layers[jlay].kx) * X
                            + (self.layers[jlay].G[j][1] + self.layers[jlay].ky) * Y
                        )
                    ),
                )
            Emx_l.append(Emx)
            Emy_l.append(Emy)
        Em = np.add(
            ul
            * np.exp(
                (0.0 + 2.0j)
                * np.pi
                * self.layers[jlay].k0
                * self.layers[jlay].gamma
                * z
                * self.d[jlay]
            ),
            dl
            * np.exp(
                -(0.0 + 2.0j)
                * np.pi
                * self.layers[jlay].k0
                * self.layers[jlay].gamma
                * z
                * self.d[jlay]
            ),
        )
        Ex, Ey = np.zeros(np.shape(X), complex), np.zeros(np.shape(X), complex)
        for i in range(2 * self.NPW):
            Ex = np.add(Ex, Em[i] * Emx_l[i])
            Ey = np.add(Ey, Em[i] * Emy_l[i])
        f = open(filename, "w")
        # f.write('#    y           yt         z           ReEx        ImEx        AbsEx       ReEy        ImEy        AbsEy \n')
        f.write(
            "#    x           xt         y           yt          ReEx        ImEx        AbsEx       ReEy        ImEy        AbsEy \n"
        )
        for i in range(np.shape(Ex)[0]):
            for j in range(np.shape(Ex)[1]):
                f.write(
                    10
                    * "%15.6e"
                    % (
                        x[i],
                        xl[i],
                        y[j] * self.Nyx,
                        yl[j] * self.Nyx,
                        Ex[i, j].real,
                        Ex[i, j].imag,
                        abs(Ex[i, j]),
                        Ey[i, j].real,
                        Ey[i, j].imag,
                        abs(Ey[i, j]),
                    )
                    + "\n"
                )
            f.write("\n")
        f.close()

    def create_input(self, dic):
        u = np.zeros((2 * self.NPW), complex)
        for i in dic:
            u[np.argsort(self.layers[0].W)[-i]] = dic[i]
        return u

    def plot_E_general(
        self, u, d=None, dz=0.01, pdf=None, N=100, y=0.0, func=np.real, s=1
    ):
        u1, d2 = np.zeros((2 * self.NPW), complex), np.zeros((2 * self.NPW), complex)
        # u1[np.argsort(self.layers[0].W)[-i]]=1.0+0.0j
        u1 = u
        if d != None:
            d2 = d
        (u2, d1) = self.S.output(u1, d2)
        x = np.linspace(-s * 0.5, s * 0.5, s * N)
        ind = range(2 * self.NPW)
        [X, I] = np.meshgrid(x, ind)
        Ex, Ey = [], []
        # first layer
        lay = self.layers[0]
        d = self.d[0]
        Emx, Emy = np.zeros(np.shape(X), complex), np.zeros(np.shape(X), complex)
        for j in range(self.NPW):
            Emx = np.add(
                Emx,
                lay.V[j, I]
                * np.exp(
                    (0.0 + 2.0j)
                    * np.pi
                    * ((lay.G[j][0] + lay.kx) * X + (lay.G[j][1] + lay.ky) * y)
                ),
            )
            Emy = np.add(
                Emy,
                lay.V[j + self.NPW, I]
                * np.exp(
                    (0.0 + 2.0j)
                    * np.pi
                    * ((lay.G[j][0] + lay.kx) * X + (lay.G[j][1] + lay.ky) * y)
                ),
            )
        for z in np.arange(-d, 0.0, dz):
            Em = np.add(
                u1 * np.exp((0.0 + 2.0j) * np.pi * lay.k0 * lay.gamma * z),
                d1 * np.exp(-(0.0 + 2.0j) * np.pi * lay.k0 * lay.gamma * z),
            )
            Ex.append(np.dot(Em, Emx))
            Ey.append(np.dot(Em, Emy))
        # intermediate layers
        S1 = copy.deepcopy(self.int_matrices[self.int_list.index(self.interfaces[0])])
        for i in range(1, self.N - 1):
            S2 = S_matrix(S1.N)
            for l in range(i, self.N - 1):
                S2.add_uniform(self.layers[l], self.d[l])
                S2.add(self.int_matrices[self.int_list.index(self.interfaces[l])])
            (ul, dl) = S1.int_f_tot(S2, u1, d2)
            Emx, Emy = np.zeros(np.shape(X), complex), np.zeros(np.shape(X), complex)
            for j in range(self.NPW):
                Emx = np.add(
                    Emx,
                    self.layers[i].V[j, I]
                    * np.exp(
                        (0.0 + 2.0j)
                        * np.pi
                        * (
                            (self.layers[i].G[j][0] + self.layers[i].kx) * X
                            + (self.layers[i].G[j][1] + self.layers[i].ky) * y
                        )
                    ),
                )
                Emy = np.add(
                    Emy,
                    self.layers[i].V[j + self.NPW, I]
                    * np.exp(
                        (0.0 + 2.0j)
                        * np.pi
                        * (
                            (self.layers[i].G[j][0] + self.layers[i].kx) * X
                            + (self.layers[i].G[j][1] + self.layers[i].ky) * y
                        )
                    ),
                )
            for z in np.arange(0.0, self.d[i], dz):
                Em = np.add(
                    ul
                    * np.exp(
                        (0.0 + 2.0j)
                        * np.pi
                        * self.layers[i].k0
                        * self.layers[i].gamma
                        * z
                    ),
                    dl
                    * np.exp(
                        -(0.0 + 2.0j)
                        * np.pi
                        * self.layers[i].k0
                        * self.layers[i].gamma
                        * z
                    ),
                )
                Ex.append(np.dot(Em, Emx))
                Ey.append(np.dot(Em, Emy))
            S1.add_uniform(self.layers[i], self.d[i])
            S1.add(self.int_matrices[self.int_list.index(self.interfaces[i])])
        # last layer
        lay = self.layers[-1]
        d = self.d[-1]
        Emx, Emy = np.zeros(np.shape(X), complex), np.zeros(np.shape(X), complex)
        for j in range(self.NPW):
            Emx = np.add(
                Emx,
                lay.V[j, I]
                * np.exp(
                    (0.0 + 2.0j)
                    * np.pi
                    * ((lay.G[j][0] + lay.kx) * X + (lay.G[j][1] + lay.ky) * y)
                ),
            )
            Emy = np.add(
                Emy,
                lay.V[j + self.NPW, I]
                * np.exp(
                    (0.0 + 2.0j)
                    * np.pi
                    * ((lay.G[j][0] + lay.kx) * X + (lay.G[j][1] + lay.ky) * y)
                ),
            )
        for z in np.arange(0.0, d, dz):
            Em = np.add(
                u2 * np.exp((0.0 + 2.0j) * np.pi * lay.k0 * lay.gamma * z),
                d2 * np.exp(-(0.0 + 2.0j) * np.pi * lay.k0 * lay.gamma * z),
            )
            Ex.append(np.dot(Em, Emx))
            Ey.append(np.dot(Em, Emy))
        if pdf == None:
            out = PdfPages("E.pdf")
        else:
            out = pdf
        plt.figure()
        plt.subplot(211)
        plt.title("Ex")
        plt.imshow(
            func(Ex).T, origin="lower", extent=[0.0, sum(self.d), -0.5, 0.5], cmap="jet"
        )
        plt.colorbar()
        plt.subplot(212)
        plt.title("Ey")
        plt.imshow(
            func(Ey).T, origin="lower", extent=[0.0, sum(self.d), -0.5, 0.5], cmap="jet"
        )
        plt.colorbar()
        # plt.savefig('field.png',dpi=900)
        out.savefig(dpi=900)
        plt.close()
        if pdf == None:
            out.close()
        return None

    def line_E(self, i, x=0.0, y=0.0, dz=0.01):
        u1, d2 = np.zeros((2 * self.NPW), complex), np.zeros((2 * self.NPW), complex)
        u1[np.argsort(self.layers[0].W)[-i]] = 1.0 + 0.0j
        (u2, d1) = self.S.output(u1, d2)
        ind = range(2 * self.NPW)
        I = ind
        X = x
        Ex, Ey, zl = [], [], []
        D = np.cumsum(self.d)
        # first layer
        lay = self.layers[0]
        d = self.d[0]
        Emx, Emy = np.zeros(np.shape(X), complex), np.zeros(np.shape(X), complex)
        for j in range(self.NPW):
            Emx = np.add(
                Emx,
                lay.V[j, I]
                * np.exp(
                    (0.0 + 2.0j)
                    * np.pi
                    * ((lay.G[j][0] + lay.kx) * X + (lay.G[j][1] + lay.ky) * y)
                ),
            )
            Emy = np.add(
                Emy,
                lay.V[j + self.NPW, I]
                * np.exp(
                    (0.0 + 2.0j)
                    * np.pi
                    * ((lay.G[j][0] + lay.kx) * X + (lay.G[j][1] + lay.ky) * y)
                ),
            )
        for z in np.arange(-d, 0.0, dz):
            Em = np.add(
                u1 * np.exp((0.0 + 2.0j) * np.pi * lay.k0 * lay.gamma * z),
                d1 * np.exp(-(0.0 + 2.0j) * np.pi * lay.k0 * lay.gamma * z),
            )
            Ex.append(np.dot(Em, Emx))
            Ey.append(np.dot(Em, Emy))
            zl.append(D[0] + z)
        # intermediate layers
        S1 = copy.deepcopy(self.int_matrices[self.int_list.index(self.interfaces[0])])
        for i in range(1, self.N - 1):
            S2 = S_matrix(S1.N)
            for l in range(i, self.N - 1):
                S2.add_uniform(self.layers[l], self.d[l])
                S2.add(self.int_matrices[self.int_list.index(self.interfaces[l])])
            (ul, dl) = S1.int_f(S2, u1)
            Emx, Emy = np.zeros(np.shape(X), complex), np.zeros(np.shape(X), complex)
            for j in range(self.NPW):
                Emx = np.add(
                    Emx,
                    self.layers[i].V[j, I]
                    * np.exp(
                        (0.0 + 2.0j)
                        * np.pi
                        * (
                            (self.layers[i].G[j][0] + self.layers[i].kx) * X
                            + (self.layers[i].G[j][1] + self.layers[i].ky) * y
                        )
                    ),
                )
                Emy = np.add(
                    Emy,
                    self.layers[i].V[j + self.NPW, I]
                    * np.exp(
                        (0.0 + 2.0j)
                        * np.pi
                        * (
                            (self.layers[i].G[j][0] + self.layers[i].kx) * X
                            + (self.layers[i].G[j][1] + self.layers[i].ky) * y
                        )
                    ),
                )
            for z in np.arange(0.0, self.d[i], dz):
                Em = np.add(
                    ul
                    * np.exp(
                        (0.0 + 2.0j)
                        * np.pi
                        * self.layers[i].k0
                        * self.layers[i].gamma
                        * z
                    ),
                    dl
                    * np.exp(
                        -(0.0 + 2.0j)
                        * np.pi
                        * self.layers[i].k0
                        * self.layers[i].gamma
                        * z
                    ),
                )
                Ex.append(np.dot(Em, Emx))
                Ey.append(np.dot(Em, Emy))
                zl.append(D[i - 1] + z)
            S1.add_uniform(self.layers[i], self.d[i])
            S1.add(self.int_matrices[self.int_list.index(self.interfaces[i])])
        # last layer
        lay = self.layers[-1]
        d = self.d[-1]
        Emx, Emy = np.zeros(np.shape(X), complex), np.zeros(np.shape(X), complex)
        for j in range(self.NPW):
            Emx = np.add(
                Emx,
                lay.V[j, I]
                * np.exp(
                    (0.0 + 2.0j)
                    * np.pi
                    * ((lay.G[j][0] + lay.kx) * X + (lay.G[j][1] + lay.ky) * y)
                ),
            )
            Emy = np.add(
                Emy,
                lay.V[j + self.NPW, I]
                * np.exp(
                    (0.0 + 2.0j)
                    * np.pi
                    * ((lay.G[j][0] + lay.kx) * X + (lay.G[j][1] + lay.ky) * y)
                ),
            )
        for z in np.arange(0.0, d, dz):
            Em = np.add(
                u2 * np.exp((0.0 + 2.0j) * np.pi * lay.k0 * lay.gamma * z),
                d2 * np.exp(-(0.0 + 2.0j) * np.pi * lay.k0 * lay.gamma * z),
            )
            Ex.append(np.dot(Em, Emx))
            Ey.append(np.dot(Em, Emy))
            zl.append(D[-2] + z)

        return [np.array(zl), np.array(Ex), np.array(Ey)]

    def inspect(self, st="", details="no"):
        att = sub.get_user_attributes(self)
        print(st)
        print(22 * "_")
        print("| INT argument")
        for i in att:
            if type(i[1]) is int:
                print("|%10s%10s" % (i[0], str(i[1])))
        print("| Float argument")
        for i in att:
            if type(i[1]) is float:
                print("|%10s%10s" % (i[0], str(i[1])))
        for i in att:
            if type(i[1]) is np.float64:
                print("|%10s%10s" % (i[0], str(i[1])))
        print("| BOOL argument")
        for i in att:
            if type(i[1]) is bool:
                print("|%10s%10s" % (i[0], str(i[1])))
        print("| Array argument")
        for i in att:
            if type(i[1]) is np.ndarray:
                print("|%10s%10s" % (i[0], str(np.shape(i[1]))))
        print("| List argument")
        for i in att:
            if type(i[1]) is list:
                print("|%12s%8s" % (i[0], str(len(i[1]))))
        print("")
        try:
            print("lay list:")
            for s in self.lay_list:
                print(s)

            print("layers:")
            for s in self.layers:
                print(s)

            print("int_list:")
            for s in self.int_list:
                print(s)

            print("interfaces:")
            for s in self.interfaces:
                print(s)
        except AttributeError:
            print("No list yet, call conut_interface before inspect")
