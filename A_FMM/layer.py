import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import pandas as pd

import A_FMM
import A_FMM.sub_sm as sub
from A_FMM.creator import Creator
from matplotlib.backends.backend_pdf import PdfPages
from A_FMM.scattering import S_matrix
import copy


class Layer:
    """Class for the definition of a single layer"""

    def __init__(self, Nx: int, Ny: int, creator: Creator, Nyx: float = 1.0):
        """Creator

        Args:
            Nx (int): truncation order in x direction
            Ny (int): truncation order in y direction
            Nyx (float): ratio between the cell's dimension in y and x (ay/ax)
        """
        self.Nx = Nx
        self.Ny = Ny
        self.NPW = (2 * Nx + 1) * (2 * Ny + 1)
        self.G = sub.createG(self.Nx, self.Ny)
        self.G_inv = {v: k for k, v in self.G.items()}
        self.D = len(self.G)
        self.creator = copy.deepcopy(creator)
        self.Nyx = Nyx

        self.FOUP = self.__create_eps()
        self.INV = linalg.inv(self.FOUP)

        self.EPS1 = sub.fou_xy(
            self.Nx,
            self.Ny,
            self.G,
            self.creator.x_list,
            self.creator.y_list,
            self.creator.eps_lists,
        )
        self.EPS2 = sub.fou_yx(
            self.Nx,
            self.Ny,
            self.G,
            self.creator.x_list,
            self.creator.y_list,
            self.creator.eps_lists,
        )

        self.TX = False
        self.TY = False

    def __create_eps(self):
        nx = 2 * self.Nx
        ny = 2 * self.Ny
        mx = 4 * self.Nx + 1
        my = 4 * self.Ny + 1
        fourier_transform = np.zeros((mx, my), dtype=complex)
        x_list = self.creator.x_list
        y_list = self.creator.y_list
        eps_lists = self.creator.eps_lists
        G = self.G
        for i in range(mx):
            for j in range(my):
                fourier_transform[i, j] = sub.fou(
                    (i + nx) % mx - nx, (j + ny) % my - ny, x_list, y_list, eps_lists
                )
        D = len(G)
        F = np.zeros((D, D), complex)
        for i, (gx1, gy1) in G.items():
            for j, (gx2, gy2) in G.items():
                F[i, j] = fourier_transform[gx1 - gx2, gy1 - gy2]
        return F

    def inspect(self, st=""):
        """Function for inspectig the attributes of a layer object

        Args:
            st (string): string to print before the inspection for identification
        """
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
        print("")

    def eps_plot(self, pdf=None, N=200, s=1):
        """Function for plotting the dielectric consstat rebuit from plane wave expansion

        Args:
            pdf (string or PdfPages): file for printing the the epsilon
                if a PdfPages object, the page is appended to the pdf
                if string, a pdf with that name is created
            N (int): number of points
            s (float): number of cell replicas to display (default 1)
        """
        [X, Y] = np.meshgrid(
            np.linspace(-s * 0.5, s * 0.5, s * N),
            np.linspace(-s * 0.5, s * 0.5, int(s * N * self.Nyx)),
        )
        EPS = np.zeros((N, N), complex)
        #        for i in range(self.D):
        #            EPS+=sub.fou(self.G[i][0],self.G[i][1],self.creator.x_list,self.creator.y_list,self.creator.eps_lists)*np.exp(-(0+2j)*np.pi*(self.G[i][0]*X+self.G[i][1]*Y))
        EPS = sum(
            [
                sub.fou(
                    self.G[i][0],
                    self.G[i][1],
                    self.creator.x_list,
                    self.creator.y_list,
                    self.creator.eps_lists,
                )
                * np.exp((0 + 2j) * np.pi * (self.G[i][0] * X + self.G[i][1] * Y))
                for i in range(self.D)
            ]
        )
        plt.figure()
        # plt.imshow(np.real(EPS),aspect='auto',extent=[-s*0.5,s*0.5,-self.Nyx*s*0.5,self.Nyx*s*0.5])
        plt.imshow(
            np.real(EPS),
            extent=[-s * 0.5, s * 0.5, -self.Nyx * s * 0.5, self.Nyx * s * 0.5],
            origin="lower",
        )
        plt.colorbar()
        if pdf == None:
            plt.show()
        elif isinstance(pdf, PdfPages):
            pdf.savefig()
        else:
            a = PdfPages(pdf + ".pdf")
            a.savefig()
            a.close()
        plt.close()

    def transform(self, ex: float = 0, ey: float = 0, complex_transform: bool = False):
        """Function for adding the real coordinate transfomr to the layer

        Note: for no mapping, set the width to 0

        Args:
            ex (float): relative width of the unmapped region in x direction. Default is 0 (no mapping)
            ey (float): relative width of the unmapped region in y direction. Default is 0 (no mapping)
            complex_transform (bool): False for real transform (default), True for complex one.
        """
        if complex_transform:
            transform_function = sub.fou_complex_t
        else:
            transform_function = sub.fou_t

        if ex != 0.0:
            self.TX = True
            self.ex = ex
            self.FX = np.zeros((self.D, self.D), complex)
            nx = 2 * self.Nx
            mx = 4 * self.Nx + 1
            F = [transform_function((i + nx) % mx - nx, ex) for i in range(mx)]
            for i, (gx1, gy1) in self.G.items():
                for j, (gx2, gy2) in self.G.items():
                    if gy1 != gy2:
                        continue
                    self.FX[i, j] = F[gx1 - gx2]
        else:
            self.FX = None

        if ey != 0.0:
            self.TY = True
            self.ey = ey
            self.FY = np.zeros((self.D, self.D), complex)
            ny = 2 * self.Ny
            my = 4 * self.Ny + 1
            F = [transform_function((i + ny) % my - ny, ey) for i in range(my)]
            for i, (gx1, gy1) in self.G.items():
                for j, (gx2, gy2) in self.G.items():
                    if gx1 != gx2:
                        continue
                    self.FY[i, j] = F[gy1 - gy2]
        else:
            self.FY = None

        return self.FX, self.FY

    def add_transform_matrix(
        self,
        ex: float = 0.0,
        FX: np.ndarray = None,
        ey: float = 0.0,
        FY: np.ndarray = None,
    ):
        """Function for adding matrix of a coordinate transform

        Args:
            ex (float): relative width of the unmapped region in x direction. Default is 0. This is only for keeping track of the value, as it has no effect on the transformation.
            FX (ndarray): FX matrix of the coordinate trasnformation
            ey (float): relative width of the unmapped region in y direction. Default is 0. This is only for keeping track of the value, as it has no effect on the transformation.
            FY (ndarray): FY matrix of the coordinate trasnformation
        """
        if ex != 0:
            self.TX = True
            self.ex = ex
            self.FX = FX
        else:
            self.FX = None
        if ey != 0:
            self.TY = True
            self.ey = ey
            self.FY = FY
        else:
            self.FY = None

    def mode(self, k0: float, kx: float = 0.0, ky: float = 0.0):
        """Calculates the eighenmode of the layer

        Args:
            k0 (float): Vacuum wavevector
            kx (float): Wavevector in the x direction
            ky (float): Wavevector in the y direction
        """
        self.k0 = k0
        self.kx = kx
        self.ky = ky
        (k1, k2) = sub.createK(self.G, k0, kx=kx, ky=ky, Nyx=self.Nyx)
        if self.TX:
            k1 = np.dot(self.FX, k1)
        if self.TY:
            k2 = np.dot(self.FY, k2)
        self.GH, self.M = sub.create_2order_new(
            self.D, k1, k2, self.INV, self.EPS1, self.EPS2
        )
        [self.W, self.V] = linalg.eig(self.M)
        self.gamma = np.sqrt(self.W) * np.sign(np.angle(self.W) + 0.5 * np.pi)
        if np.any(np.real(self.gamma) + np.imag(self.gamma) <= 0.0):
            print("Warining: wrong complex root")
        if np.any(np.abs(self.gamma) <= 0.0):
            print("Warining: gamma=0")
        self.VH = np.dot(self.GH, self.V / self.gamma)

    def clear(self):
        """Removes data created in mode method"""
        self.VH = None
        self.M = None
        self.GH = None
        self.W = None
        self.V = None
        self.gamma = None

    def get_index(self, ordered: bool = True) -> np.ndarray:
        """Returns the effective idexes of the modes

        Args:
            ordered (bool): if True (default) the modes are ordered by decreasing effective index
        """
        if ordered:
            Neff = np.sort(self.gamma)[::-1]
        else:
            Neff = self.gamma
        return Neff

    def mat_plot(self, name: str):
        """Plot the absolute values of the fourier trasnsform matrices

        Args:
            name (str): name of the pdf file for plotting
            N (int): number of points for plotting the epsilon
            s (float): number pf relicas of the cell to plot. Default is 1.
        """
        with PdfPages(name + ".pdf") as save:
            for attr in ["FOUP", "EPS1", "EPS2", "INV", "FX", "FY"]:
                try:
                    to_plot = getattr(self, attr)
                    plt.figure()
                    plt.title(attr)
                    plt.imshow(np.abs(to_plot), aspect="auto", interpolation="nearest")
                    plt.colorbar()
                    save.savefig()
                    plt.close()
                except AttributeError:
                    pass

    def plot_Ham(self, pdf: PdfPages) -> None:
        """Plot the matrix of the eigenvalue problem

        Args:
            pdf (PdfPages): pdf object to be used to plot.

        Returns:
            None.

        """
        plt.figure()
        plt.title("k0:%5.3f kx:%5.3f ky:%5.3f" % (self.k0, self.kx, self.ky))
        plt.imshow(np.abs(np.abs(self.M)), aspect="auto", interpolation="nearest")
        plt.colorbar()
        pdf.savefig()
        plt.close()

    def _process_xy(self, x: np.ndarray, y: np.ndarray) -> tuple:
        """Transform the x and y coordinates between the real and computational space

        Args:
            x (ndarray): array of x coordinates in the real space
            y (ndarray): array of y coordinates in the real space

        Returns: tuple of numpy.ndarray containing:
            - ndarray: array of x coordinates in the computational space
            - ndarray: array of y coordinates in the computational space

        """
        if self.TX:
            x = sub.t_inv(x, self.ex)
        if self.TY:
            y = sub.t_inv(y, self.ey)
        return x, y

    def calculate_epsilon(
        self, x: np.ndarray=0.0, y: np.ndarray=0.0, z: np.ndarray=0.0
    ) -> dict[str, np.ndarray]:
        """Return epsilon given the coordinates

        The epsilon returned here is the one reconstructed from the Fourier transform.
        The epsilon is reconstructed on the meshgrid of x,y, and z.

        Args:
            x (array_like): x coordinates (1D array).
            y (array_like): y coordinates (1D array).
            z (array_like): z coordinates (1D array).

        Returns:
            ndarray : Epsilon value at coordinates. Shape of ndarray is the same as x,y, and z.

        Raises:
            ValueError:
                if x,y,z have different shapes
        """

        x, y = self._process_xy(x, y)
        eps = self.FOUP[:, self.D // 2]
        gx, gy = zip(*[g for i, g in self.G.items()])
        gx, gy = np.asarray(gx), np.asarray(gy)
        xp, yp, gxp = np.meshgrid(x, y, gx, indexing='ij')
        xp, yp, gyp = np.meshgrid(x, y, gy, indexing='ij')
        eps_p = np.dot(
            np.exp((0 + 2j) * np.pi * (gxp * xp + gyp * yp)),
            eps
        )
        shape = np.shape(eps_p)
        EPS, _ = np.meshgrid(eps_p, z, indexing='ij')
        EPS = EPS.reshape(*shape, -1)
        x,y,z = np.meshgrid(x,y,z, indexing='ij')
        eps = {
            'x' : x,
            'y' : y,
            'z' : z,
            'eps' : EPS,
        }
        return eps


    @staticmethod
    def _filter_componets(components: list=None) -> list:
        """
        Checks if the fileds components list contains only allowed ones
        """
        if components is None:
            return ["Ex", "Ey", "Hx", "Hy"]
        for comp in components:
            if comp not in ["Ex", "Ey", "Hx", "Hy"]:
                raise ValueError(
                    f"Field component f{comp} not available. Only Ex, Ey, Hx, or Hy are allowed"
                )
        return components

    @staticmethod
    def _check_array_shapes(u: np.ndarray, d: np.ndarray) -> None:
        """
        Chekcs that the modal amplitudea arrays and the coordinates arrays have consistent shapes
        """
        if np.shape(u) != np.shape(d):
            raise ValueError(
                f"Shape of u different from shape of d {np.shape(u)}!={np.shape(d)}"
            )



    def calculate_field_old(
        self,
        u: np.ndarray,
        d: np.ndarray = None,
        x: np.ndarray=0,
        y: np.ndarray=0,
        z: np.ndarray=0,
        components: list = None,
    ) -> dict:
        """Return field given modal coefficient and coordinates

        Coordinates arrays must be 1D. Fields are returned on a meshgrid of the input coordinates.
        Older version. Slower, but may require less memory.

        Args:
            u (array_like): coefficient of forward propagating modes.
            d (array_like, optional): coefficient of backward propagating modes.
                Default to None: no backward propagation is assumed.
            x (array_like): x coordinates.
            y (array_like): y coordinates.
            z (array_like): z coordinates.
            components (list of str, optional): field components to calculate.
                Default to None: all components ('Ex', 'Ey', 'Hx', 'Hy') are calculated.

        Returns:
            dict of ndarray : Desired field components. Shape of ndarray is the same as x,y, and z.

        Raises:
            ValueError:
                if other component than 'Ex', 'Ey', 'Hx', or 'Hy' is requested.

        """
        components = self._filter_componets(components)
        d = np.zeros_like(u, dtype=complex) if d is None else d
        self._check_array_shapes(u,d)

        x, y = self._process_xy(x, y)
        x,y,z = np.meshgrid(x,y,z, indexing='ij')
        field = {
            'x' : x,
            'y' : y,
            'z' : z,
        }
        field.update({comp: np.zeros_like(x, dtype=complex) for comp in components})
        for i, (uu, dd, n) in enumerate(zip(u, d, self.gamma)):
            if uu == 0.0 and dd == 0.0:
                continue
            field_tmp = {comp: np.zeros_like(x, dtype=complex) for comp in components}
            for j, (gx, gy) in self.G.items():
                [WEx, WEy] = np.split(self.V[:, i], 2)
                [WHx, WHy] = np.split(self.VH[:, i], 2)
                EXP = np.exp(
                    (0 + 2j) * np.pi * ((gx + self.kx) * x + (gy + self.ky) * y)
                )
                for comp in components:
                    sign = 1.0 if comp[0] == "E" else -1.0
                    coeff = uu * np.exp(2.0j * np.pi * self.k0 * n * z) + sign * dd * np.exp(
                        -2.0j * np.pi * self.k0 * n * z
                    )
                    field_tmp[comp] = (
                        field_tmp[comp] + coeff * eval(f"W{comp}")[j] * EXP
                    )
            for comp in components:
                field[comp] = field[comp] + field_tmp[comp]
        return field


    def calculate_field(
        self,
        u: np.ndarray,
        d: np.ndarray = None,
        x: np.ndarray=0,
        y: np.ndarray=0,
        z: np.ndarray=0,
        components: list = None,
    ) -> dict:
        """Return field given modal coefficient and coordinates

        Coordinates arrays must be 1D. Fields are returned on a meshgrid of the input coordinates.

        Args:
            u (array_like): coefficient of forward propagating modes.
            d (array_like, optional): coefficient of backward propagating modes.
                Default to None: no backward propagation is assumed.
            x (array_like): x coordinates.
            y (array_like): y coordinates.
            z (array_like): z coordinates.
            components (list of str, optional): field components to calculate.
                Default to None: all components ('Ex', 'Ey', 'Hx', 'Hy') are calculated.

        Returns:
            dict of ndarray : Desired field components. Shape of ndarray is the same as x,y, and z.

        Raises:
            ValueError:
                if other component than 'Ex', 'Ey', 'Hx', or 'Hy' is requested.

        """
        components = self._filter_componets(components)
        d = np.zeros_like(u, dtype=complex) if d is None else d
        self._check_array_shapes(u,d)
        X,Y,Z = np.meshgrid(x,y,z, indexing='ij')
        field = {
            'x': X,
            'y': Y,
            'z': Z,
        }
        Gx = [gs[0] for (i, gs) in self.G.items()]
        Gy = [gs[1] for (i, gs) in self.G.items()]
        u,d = np.asarray(u), np.asarray(d)
        ind = [i for i, (uu,dd) in enumerate(zip(u,d)) if uu!=0.0 or dd!=0.0 ]
        u = u[ind]
        d = d[ind]
        WEx, WEy = np.split(self.V, 2, axis=0)
        WHx, WHy = np.split(self.VH, 2, axis=0)
        W = {
            'Ex': WEx[:,ind],
            'Ey': WEy[:,ind],
            'Hx': WHx[:,ind],
            'Hy': WHy[:,ind],
        }
        X, Y, Gx = np.meshgrid(x, y, Gx, indexing = 'ij')
        X, Y, Gy = np.meshgrid(x, y, Gy, indexing = 'ij')
        EXP = np.exp(
            2.0j * np.pi * ((Gx + self.kx) * X + (Gy + self.ky) * Y)
        )
        u, Z = np.meshgrid(u, z, indexing='ij')
        d, Z = np.meshgrid(d, z, indexing='ij')
        n, Z = np.meshgrid(self.gamma[ind], z, indexing='ij')
        z_exp = 2.0j * np.pi * self.k0 * n * Z
        coeff_u = u * np.exp(z_exp)
        coeff_d = d * np.exp(-z_exp)
        for comp in components:
            coeff = coeff_u + coeff_d if 'E' in comp else coeff_u - coeff_d
            EXPV = np.dot(EXP, W[comp])
            field[comp] = np.dot(EXPV, coeff)
        return field


    def get_modal_field(
        self, i: int, x: float = 0.0, y: float = 0.0, components: list = None
    ) -> dict:
        """Returns modal field profile

        Args:
            i (int): index of the mode.
            x (float or array_like): x coordinate for the field calculation
            y (float or array_like): y coordinate for the field calculation
            components (list of str, optional): field components to calculate.
                Default to None: all components ('Ex', 'Ey', 'Hx', 'Hy') are calculated.

        Returns:
            DataFrame: modal field
        """
        x, y, z = np.meshgrid(x, y, [0.0], indexing="ij")
        u = self.create_input({i: 1.0})
        data = {
            "x": np.squeeze(x),
            "y": np.squeeze(y),
        }
        field = self.calculate_field(x, y, z, u, components=components)
        for k, v in field.items():
            data[k] = np.squeeze(v)
        return data


    def get_P_norm(self):
        """Creates array of single mode Poynting vector components.

        It is stored in the P_norm attribute

        Returns:
            None.

        """
        [VEx, VEy] = np.split(self.V, 2)
        [VHx, VHy] = np.split(self.VH, 2)
        self.P_norm = np.sum(VEx * np.conj(VHy) - VEy * np.conj(VHx), 0).real


    def get_Poynting_single(self, i: int, u: np.ndarray, ordered: bool = True) -> float:
        """Return the Poyinting vector of a single mode given the modal expansion in the layer


        Args:
            i (int): Index of the mode.
            u (1darray): Array of modal coefficient.
            ordered (TYPE, optional): Regulates how mode are ordered. If True, they are ordered for decreasing effective index. If Flase, the order is whatever is returned by the diagonalization routine. Defaults to True.

        Returns:
            TYPE: DESCRIPTION.

        """
        if ordered:
            j = np.argsort(self.W)[-i - 1]
        else:
            j = i
        self.get_Poyinting_norm()
        return self.PP_norm[j, j].real * np.abs(u[j]) ** 2.0


    def get_Poyinting_norm(self):
        """Calculates the normalization matrix for the Poyinting vector calculations


        Returns:
            None.

        """
        [VEx, VEy] = np.split(self.V, 2)
        [VHx, VHy] = np.conj(np.split(self.VH, 2))
        # old version (working)
        # self.PP_norm=np.zeros((2*self.D,2*self.D),dtype=complex)
        # for i in range(self.D):
        #    VEX,VHY=np.meshgrid(VEx[i,:],VHy[i,:])
        #    VEY,VHX=np.meshgrid(VEy[i,:],VHx[i,:])
        #    P1=np.multiply(VEX,VHY)
        #    P2=-np.multiply(VEY,VHX)
        #    P=np.add(P1,P2)
        #    self.PP_norm=np.add(self.PP_norm,P)
        # print self.PP_norm
        # new version. should be equivalent bit faster
        P1 = np.dot(np.transpose(VEx), VHy)
        P2 = np.dot(np.transpose(VEy), VHx)
        self.PP_norm = np.add(P1, -P2)

    def get_Poynting(self, u: np.ndarray, d: np.ndarray = None):
        """Calculates total Poynting vector in the layer given arrays of modal expansion


        Args:
            u (1darray): Modal expansion of forward propagating modes.
            d (1darray, optional): Modal expansion of backward propagating modes. Defaults to None.

        Returns:
            TYPE: DESCRIPTION.

        """
        if d is None:
            d = np.zeros(2 * self.D, dtype=complex)
        # try:
        #    self.PP_norm
        # except AttributeError:
        #    self.get_Poyinting_norm()
        self.get_Poyinting_norm()
        Cn = np.add(u, d)
        Cnp = np.add(u, -d)
        [Cn, Cnp] = np.meshgrid(Cn, np.conj(Cnp))
        C = np.multiply(Cn, Cnp)
        PP = np.multiply(C, self.PP_norm)
        return np.sum(PP).real

    def T_interface(self, lay) -> np.ndarray:
        """Builds the Transfer matrix of the interface with another layer

        Args:
            lay (Layer): Layer toward which to calculate the scattering matrix.

        Returns:
            T (2darray): Interface scattering matrix.

        """
        T1 = np.dot(linalg.inv(lay.V), self.V)
        T2 = np.dot(linalg.inv(lay.VH), self.VH)
        T11 = 0.5 * (T1 + T2)
        T12 = 0.5 * (T1 - T2)
        T21 = 0.5 * (T1 - T2)
        T22 = 0.5 * (T1 + T2)
        T = np.vstack([np.hstack([T11, T12]), np.hstack([T21, T22])])
        return T

    def T_prop(self, d: float) -> np.ndarray:
        """Build the propagation Transfer matrix of the layer

        Args:
            d (float): Thickness of the layer.

        Returns:
            T (2darray): Propagation Transfer matrix.

        """
        I1 = np.diag(np.exp((0 + 1j) * self.k0 * self.gamma * d))
        I2 = np.diag(np.exp(-(0 + 1j) * self.k0 * self.gamma * d))
        I = np.zeros((2 * self.D, 2 * self.D), complex)
        T = np.vstack([np.hstack([I1, I]), np.hstack([I, I2])])
        return T

    # newer version, should be faster
    def interface(self, lay) -> S_matrix:
        """Builds the Scattering matrix of the interface with another layer

        Args:
            lay (Layer): Layer toward which to calculate the scattering matrix.

        Returns:
            S (S_matrix): Interface scattering matrix.

        """
        S = S_matrix(2 * self.D)
        T1 = np.dot(linalg.inv(lay.V), self.V)
        T2 = np.dot(linalg.inv(lay.VH), self.VH)
        T11 = 0.5 * (T1 + T2)
        T12 = 0.5 * (T1 - T2)
        # T21= 0.5*(T1 - T2)
        # T22= 0.5*(T1 + T2)
        # T=np.vstack([np.hstack([T11,T12]),np.hstack([T21,T22])])
        Tm = linalg.inv(T11)
        S.S11 = T11 - np.dot(np.dot(T12, Tm), T12)
        S.S12 = np.dot(T12, Tm)
        S.S21 = -np.dot(Tm, T12)
        S.S22 = Tm
        return S

    def get_input(
        self,
        func: callable,
        args: tuple = None,
        Nxp: int = 1024,
        Nyp: int = None,
        fileprint: str = None,
    ) -> np.ndarray:
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

        if Nyp == None:
            if self.Ny == 0:
                Nyp = 1
                y = np.array([0.0])
            else:
                Nyp = Nxp
                y = np.linspace(-0.5, 0.5, Nyp)
        else:
            y = np.linspace(-0.5, 0.5, Nyp)
        x = np.linspace(-0.5, 0.5, Nxp)
        if self.TX:
            ex = self.ex
            x = sub.t_dir(x, ex)
        if self.TY:
            ey = self.ey
            y = sub.t_dir(y, ey)

        y = y * self.Nyx

        [X, Y] = np.meshgrid(x, y, indexing="ij")
        [Fx, Fy] = func(X, Y, *args)

        Fx = np.fft.fftshift(Fx) / (Nxp * Nyp)
        Fy = np.fft.fftshift(Fy) / (Nxp * Nyp)

        FOUx = np.fft.fft2(Fx)
        FOUy = np.fft.fft2(Fy)

        Estar = np.zeros(2 * self.NPW, dtype=complex)
        for i in range(self.NPW):
            # print self.G[i][0], self.G[i][1],FOUx[self.G[i][0],self.G[i][1]]
            Estar[i] = FOUx[self.G[i][0], self.G[i][1]]
            Estar[i + self.NPW] = FOUy[self.G[i][0], self.G[i][1]]

        u = linalg.solve(self.V, Estar)
        return u

    def create_input(self, dic: dict) -> np.ndarray:
        """Creates the array of modal coefficient using a dictionary as input


        Args:
            dic (dict): Dictionary of exited modes {modal_index : modal_coeff}. Modes are ordered.

        Returns:
            u (1darray): Array of modal coefficient.

        """
        u = np.zeros((2 * self.NPW), complex)
        for i in dic:
            u[np.argsort(self.W)[-i - 1]] = dic[i]
        return u

    def get_Enorm(self):
        """Calculate field normalization


        Returns:
            None.

        """
        [VEx, VEy] = np.split(self.V, 2)
        self.ENx = np.dot(np.transpose(VEx), np.conj(VEx))
        self.ENy = np.dot(np.transpose(VEy), np.conj(VEy))

    def overlap(self, u: np.ndarray, up: np.ndarray = None):
        """EXPERIMENTAL: Calculates overlap between two fields given the modal expansion

        Args:
            u (1darray): Modal coefficient of first mode.
            up (1darray, optional): Modal coefficient of first mode. Defaults to None (up=u, namely normalization is returned).

        Returns:
            list: [tx, tx]: floats. Namely overlap in x and y polarization

        """
        if up is None:
            up = u
        try:
            self.ENx
        except AttributeError:
            self.get_Enorm()
        # print np.shape(u),np.shape(self.ENx)
        tx = np.dot(self.ENx, np.conj(up))
        ty = np.dot(self.ENy, np.conj(up))
        tx = np.dot(np.transpose(u), tx)
        ty = np.dot(np.transpose(u), ty)
        # print tx,ty
        return [tx, ty]

    def coupling(self, u: np.ndarray, up: np.ndarray) -> tuple:
        """EXPERIMENTAL: Calculates coupling between two modes given their modal exapnsion

        Args:
            u (TYPE): Modal coefficient of first mode.
            up (TYPE): Modal coefficient of second mode.

        Returns:
            list: [tx, tx]: floats. Coupling in x and y polarization.

        """
        self.get_Enorm()
        [tx1, ty1] = self.overlap(u)
        [tx2, ty2] = self.overlap(up)
        [txc, tyc] = self.overlap(u, up)
        return txc / np.sqrt(tx1 * tx2), tyc / np.sqrt(ty1 * ty2)


class Layer_ani_diag(Layer):
    """Class for the definition of a single layer anysitropic (diagonal) layer"""

    def __init__(
        self,
        Nx: int,
        Ny: int,
        creator_x: Creator,
        creator_y: Creator,
        creator_z: Creator,
        Nyx: float = 1.0,
    ):
        self.Nx = Nx
        self.Ny = Ny
        self.NPW = (2 * Nx + 1) * (2 * Ny + 1)
        self.G = sub.createG(self.Nx, self.Ny)
        self.D = len(self.G)
        self.creator = [
            copy.deepcopy(creator_x),
            copy.deepcopy(creator_y),
            copy.deepcopy(creator_z),
        ]
        self.Nyx = Nyx

        self.FOUP = sub.create_epsilon(
            self.G,
            self.creator[2].x_list,
            self.creator[2].y_list,
            self.creator[2].eps_lists,
        )
        self.INV = linalg.inv(self.FOUP)

        self.EPS1 = sub.fou_xy(
            self.Nx,
            self.Ny,
            self.G,
            self.creator[0].x_list,
            self.creator[0].y_list,
            self.creator[0].eps_lists,
        )
        self.EPS2 = sub.fou_yx(
            self.Nx,
            self.Ny,
            self.G,
            self.creator[1].x_list,
            self.creator[1].y_list,
            self.creator[1].eps_lists,
        )

        self.TX = False
        self.TY = False


class Layer_num(Layer):
    """Class for the definition of a single layer from a function defining the dielectric profile"""

    def __init__(
        self,
        Nx: int,
        Ny: int,
        func: callable,
        args: tuple = None,
        Nyx: float = 1.0,
        NX: int = 2048,
        NY: int = 2048,
    ):
        self.Nx = Nx
        self.Ny = Ny
        self.NPW = (2 * Nx + 1) * (2 * Ny + 1)
        self.G = sub.createG(self.Nx, self.Ny)
        self.D = len(self.G)
        self.Nyx = Nyx
        self.func = func
        self.args = args
        self.NX = NX
        self.NY = NY
        # print args
        args = () if args is None else args

        self.FOUP = sub.num_fou(func, args, self.G, NX, NY, self.Nyx)
        self.INV = linalg.inv(self.FOUP)

        # Still to be defined
        self.EPS1 = sub.num_fou_xy(
            self.func, self.args, self.Nx, self.Ny, self.G, NX, NY, self.Nyx
        )
        self.EPS2 = sub.num_fou_yx(
            self.func, self.args, self.Nx, self.Ny, self.G, NX, NY, self.Nyx
        )

        self.TX = False
        self.TY = False


class Layer_uniform(Layer):
    """Class for the definition of a single uniform layer"""

    def __init__(self, Nx: int, Ny: int, eps: float | complex, Nyx: float = 1.0):
        self.Nx = Nx
        self.Ny = Ny
        self.NPW = (2 * Nx + 1) * (2 * Ny + 1)
        self.G = sub.createG(self.Nx, self.Ny)
        self.D = len(self.G)
        self.Nyx = Nyx
        self.eps = eps

        self.FOUP = eps * np.identity(self.D, dtype="complex")
        self.INV = 1.0 / eps * np.identity(self.D, dtype="complex")

        # Still to be defined
        self.EPS1 = eps * np.identity(self.D, dtype="complex")
        self.EPS2 = eps * np.identity(self.D, dtype="complex")

        self.TX = False
        self.TY = False


class Layer_empty_st(Layer):
    """Class for the definition of an empy layer"""

    def __init__(self, Nx: int, Ny: int, creator: Creator, Nyx: float = 1.0):
        self.Nx = Nx
        self.Ny = Ny
        self.NPW = (2 * Nx + 1) * (2 * Ny + 1)
        self.G = sub.createG(self.Nx, self.Ny)
        self.D = len(self.G)
        self.creator = copy.deepcopy(creator)
        self.Nyx = Nyx

        self.TX = False
        self.TY = False

        self.FOUP = np.zeros((self.D, self.D), dtype=complex)
        # self.INV=np.zeros((self.D,self.D),dtype=complex)
        self.INV = linalg.inv(np.eye(self.D, dtype=complex))
        self.EPS1 = np.zeros((self.D, self.D), dtype=complex)
        self.EPS2 = np.zeros((self.D, self.D), dtype=complex)

    def fourier(self):
        """Calculates the fourier transform matrices need for the eigenvalue problem.


        Returns:
            2darray: FOUP matrix.
            2darray: INV matrix.
            2darray: EPS1 matrix.
            2darray: EPS2 matrix.

        """
        self.FOUP = sub.create_epsilon(
            self.G, self.creator.x_list, self.creator.y_list, self.creator.eps_lists
        ) * (1.0 + 0.0j)
        self.INV = linalg.inv(self.FOUP)
        self.EPS1 = sub.fou_xy(
            self.Nx,
            self.Ny,
            self.G,
            self.creator.x_list,
            self.creator.y_list,
            self.creator.eps_lists,
        ) * (1.0 + 0.0j)
        self.EPS2 = sub.fou_yx(
            self.Nx,
            self.Ny,
            self.G,
            self.creator.x_list,
            self.creator.y_list,
            self.creator.eps_lists,
        ) * (1.0 + 0.0j)
        return (self.FOUP, self.INV, self.EPS1, self.EPS2)


if __name__ == "__main__":
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    cr = Creator()
    cr.rect(12.0, 2.0, 0.5, 0.2)
    lay = Layer(15, 15, cr)
    t = np.linspace(-0.5, 0.5, 101)
    x, y, z = t, t, 0.0
    x, y, z = t, 0.0, t
    x, y, z = 0.0, t, t
    x, y, z = t, t, t
    eps = lay.calculate_epsilon(x,y,z)
    ax[0].contourf(
        eps['x'][:,:,50],
        eps['y'][:,:,50],
        eps['eps'][:,:,50],
    )
    ax[1].contourf(
        eps['x'][:,50,:],
        eps['z'][:,50,:],
        eps['eps'][:,50,:],
    )
    ax[2].contourf(
        eps['y'][50,:,:],
        eps['z'][50,:,:],
        eps['eps'][50,:,:],
    )


    plt.show()

