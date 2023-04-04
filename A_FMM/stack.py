from __future__ import annotations

import numpy as np

import A_FMM
import A_FMM.layer
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

    def __init__(self, layers: list[Layer] = None, d: list[float] = None) -> None:
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

    @property
    def total_length(self):
        return sum(self.d)

    def add_layer(self, lay: Layer, d: float) -> None:
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

    def transform(
        self, ex: float = 0, ey: float = 0, complex_transform: bool = False
    ) -> tuple[np.ndarray]:
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

    def count_interface(self) -> None:
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

    def solve(self, k0: float, kx: float = 0.0, ky: float = 0.0) -> None:
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

    def solve_serial(self, k0: float, kx: float = 0.0, ky: float = 0.0) -> None:
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

    def solve_lay(self, k0: float, kx: float = 0.0, ky: float = 0.0) -> None:
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

    def solve_S(self) -> None:
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

    def get_prop(
        self, u: np.ndarray, list_lay: list[int], d: np.ndarray = None
    ) -> dict[int, float]:
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

    def get_energybalance(self, u: np.ndarray, d: np.ndarray = None) -> tuple[float]:
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

    def get_inout(
        self, u: np.ndarray, d: np.ndarray = None
    ) -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
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

    def get_R(self, i: int, j: int, ordered: bool = True) -> float:
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

    def get_T(self, i: int, j: int, ordered: bool = True) -> float:
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

    def get_PR(self, i: int, j: int, ordered: bool = True) -> float:
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

    def get_PT(self, i: int, j: int, ordered: bool = True) -> float:
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

    def get_el(self, sel: str, i: int, j: int) -> complex:
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

    def double(self) -> None:
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

    def join(self, st2: Stack) -> None:
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

    def flip(self) -> None:
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

    def bloch_modes(self) -> np.ndarray:
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

    def loop_intermediate(self, u1: np.ndarray, d2: np.ndarray) -> tuple:
        """Generator for the intermedia modal coefficients.

        Progressively yields the forward and backward modal coefficient given the external excitation.

        Args:
            u1 (np.ndarray): forward modal coefficient of the first layer (near the interface)
            d2 ((np.ndarray):  backward modal coefficient of the last layer (near the interface)

        Yields:
            np.ndarray: forward modal amplitudes of layer
            np.ndarray: backward modal amplitudes for layer
            Layer: layer object
            float: thickness of the layer
        """
        u2, d1 = self.S.output(u1, d2)
        lay = self.layers[0]
        d = self.d[0]
        yield u1 * np.exp(-(0 + 2j) * np.pi * lay.k0 * lay.gamma * d), d1 * np.exp(
            (0 + 2j) * np.pi * lay.k0 * lay.gamma * d
        ), self.layers[0], self.d[0]
        # yield u1 , d1 , self.layers[0], self.d[0]
        S1 = copy.deepcopy(self.int_matrices[self.int_list.index(self.interfaces[0])])
        for i in range(1, self.N - 1):
            S2 = S_matrix(S1.N)
            for l in range(i, self.N - 1):
                S2.add_uniform(self.layers[l], self.d[l])
                S2.add(self.int_matrices[self.int_list.index(self.interfaces[l])])
            ul, dl = S1.int_f(S2, u1)
            yield ul, dl, self.layers[i], self.d[i]
            S1.add_uniform(self.layers[i], self.d[i])
            S1.add(self.int_matrices[self.int_list.index(self.interfaces[i])])
        yield u2, d2, self.layers[self.N - 1], self.d[self.N - 1] + 1e-6

    def calculate_epsilon(
        self,
        x: np.ndarray = 0.0,
        y: np.ndarray = 0.0,
        z: np.ndarray = 0.0,
    ) -> dict[str, np.ndarray]:
        """Returns epsilon in the stack

        Epsilon is calculated on a meshgrdi of x,y,z

        Args:
            x (np.ndarray): x coordinate (1D array)
            y (np.ndarray): y coordinate (1D array)
            z (np.ndarray): z coordinate (1D array)

        Returs:
            dict: Dictionary containing the coordinates and the epsilon
        """

        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        eps = {key: [] for key in ["x", "y", "z", "eps"]}
        cumulative_t = 0
        for lay, t in zip(self.layers + [self.layers[-1]], self.d + [np.inf]):
            ind = np.logical_and(cumulative_t <= z, z < cumulative_t + t)
            if ind.size == 0:
                continue
            zp = z[ind] - cumulative_t
            out = lay.calculate_epsilon(x, y, zp)
            out["z"] = out["z"] + cumulative_t
            for key in eps:
                eps[key].append(out[key])
            cumulative_t += t
        for key in eps:
            eps[key] = np.concatenate(eps[key], axis=-1)
        return eps

    def calculate_fields(
        self,
        u1: np.ndarray,
        d2: np.ndarray = None,
        x: np.ndarray = 0,
        y: np.ndarray = 0,
        z: np.ndarray = 0,
        components: list[str] = None,
    ) -> dict[str, np.ndarray]:
        """Returns fields in the stack

        The fields are calculated on a meshgrdi of x,y,z

        Args:
            u1 (np.ndarray): forward modal coefficient in the first layer
            d2 (np.ndarray): backward modal coefficient in the last layer
            x (np.ndarray): x coordinate (1D array)
            y (np.ndarray): y coordinate (1D array)
            z (np.ndarray): z coordinate (1D array)
            components (list): List of modal componets to be calculated. Possible are ['Ex', 'Ey', 'Hx', 'Hz'].
                Default to None (all of them).

        Returs:
            dict: Dictionary containing the coordinates and the field components
        """
        d2 = np.zeros(2 * self.NPW, dtype=complex)
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        components = Layer._filter_componets(components)
        shape = Layer._check_array_shapes(u1, d2)
        keys = ["x", "y", "z"] + components
        field = {key: [] for key in keys}
        cumulative_t = 0
        for u, d, lay, t in self.loop_intermediate(u1, d2):
            # print(u, d)
            ind = np.logical_and(cumulative_t <= z, z < cumulative_t + t)
            if ind.size == 0:
                continue
            zp = z[ind] - cumulative_t
            out = lay.calculate_field(u, d, x, y, zp, components=components)
            out["z"] = out["z"] + cumulative_t
            for key in keys:
                field[key].append(out[key])
            # plt.plot(np.abs(out['Ex']))
            # plt.show()
            # plt.plot(np.abs(field['Ex']))
            cumulative_t += t
        for key in keys:
            field[key] = np.concatenate(field[key], axis=-1)
        return field

    def inspect(self, st: str = "", details: str = "no") -> None:
        """Print some info about the Stack"""
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


if __name__ == "__main__":
    from monitor import Timer
    import pickle

    timer = Timer()

    N = 50
    cr = A_FMM.Creator()
    cr.slab(12.0, 2.0, 2.0, 0.3)
    lay1 = A_FMM.layer.Layer(N, 0, creator=cr)
    cr.slab(12.0, 2.0, 2.0, 0.1)
    lay2 = A_FMM.layer.Layer(N, 0, creator=cr)

    stack = Stack(
        10 * [lay1, lay2] + [lay1],
        [0.0] + 10 * [0.5, 0.5],
    )

    x, y, z = np.linspace(-0.5, 0.5, 101), 0.0, np.linspace(0.0, 10.0, 1000)
    eps = stack.calculate_epsilon(x, y, z)

    print(eps.keys())
    plt.contourf(
        np.squeeze(eps["z"]), np.squeeze(eps["x"]), np.squeeze(eps["eps"]), levels=41
    )
    plt.show()

    # lay1 = A_FMM.layer.Layer_uniform(0,0,2.0)
    # lay2 = A_FMM.layer.Layer_uniform(0,0,12.0)
    # stack = Stack(
    #     10 * [lay1, lay2] + [lay1],
    #     [0.0] + 10*[0.5, 0.5],
    # )
    # stack.solve(0.1)
    # x, y, z = 0.0, 0.0, np.linspace(0.0, 10.0, 1000)
    # with timer:
    #     field = stack.calculate_fields([1.0, 0.0], [0.0, 0.0], x,y,z)
    # print(timer.elapsed_time)
    # with open('test_stack_1Dfield.pkl', 'wb') as pkl:
    #     pickle.dump(field, pkl, protocol=pickle.HIGHEST_PROTOCOL)

    # layer = lay1.calculate_field([1.0, 0.0], [0.0, 0.0], x,y,z)

    # #plt.contourf(np.squeeze(field['z']), np.squeeze(field['x']), np.abs(np.squeeze(field['Ex'])), levels=41)
    # plt.show()
    # with timer:
    #     Ex, Ey = stack.plot_E(1, func=np.abs, dz = 0.01)
    # print(timer.elapsed_time)
    # Ex = np.asarray(Ex)
    # plt.show()

    # plt.plot(field['z'][0,0,:], np.abs(field['Ex'][0,0,:]))
    # plt.plot(z, np.abs(Ex[:, 50]))
    # print(np.shape(field['z']))
    # plt.show()
