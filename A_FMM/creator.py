import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import A_FMM.sub_sm as sub
from matplotlib.backends.backend_pdf import PdfPages


class Creator:
    """Class for the definition of the eps profile in the layer"""

    def __init__(self, x_list=None, y_list=None, eps_lists=None):
        """Creator

        Args:
            x_list (list):      list of floats containig the coordinates of the x boundaries
            y_list (list):      list of floats containig the coordinates of the y boundaries
            eps_lists (list):   list of list of floats containig the eps value of the squares defined by x_list and y_list
        """
        self.x_list = x_list
        self.y_list = y_list
        self.eps_lists = eps_lists

    def slow_general(self, eps_core, eps_lc, eps_uc, w, h, t, Z):
        self.x_list = np.linspace(-0.5 * w, 0.5 * w, len(Z) + 1)
        self.y_list = [-0.5, -0.5 * h, -0.5 * h + t, 0.5 * h]
        self.eps_lists = [[eps_uc, eps_lc, eps_core, eps_uc]]
        eps = [eps_uc, eps_core]
        for i in Z:
            self.eps_lists.append([eps_uc, eps_lc, eps_core, eps[i]])

    def slow_2D(self, eps_core, eps_c, w, Z):
        self.x_list = np.linspace(-0.5 * w, 0.5 * w, len(Z) + 1)
        self.y_list = [-0.5]
        self.eps_lists = [[eps_c]]
        eps = [eps_c, eps_core]
        for i in Z:
            self.eps_lists.append([eps[i]])

    def ridge(self, eps_core, eps_lc, eps_uc, w, h, t=0.0, y_offset=0.0, x_offset=0.0):
        """Rib waveguide with single layer

        Args:
            eps_core (float):   epsilon of the core
            eps_lc (floar):     epsilon of the lower cladding
            eps_up (float):     epsilon of the upper cladding
            w (float):          width of the rib (in unit of ax)
            h (float):          height of the un-etched part (in unit of ay)
            t (float):          height of the etched part (in unit of ay). Default is 0 (strip waveguide)
            x_offset (float):   offset of the center of the waveguide with respec to the center of the cell (in unit of ax). Default is 0
            y_offset (float):   offset of the etched part with resoect to the unetched one (in unit of ay). Default is 0 (etched and unetched part are aligned at the bottom)
        """
        self.x_list = [-0.5 * w + x_offset, 0.5 * w + x_offset]
        self.y_list = [
            -0.5,
            -0.5 * h,
            -0.5 * h + y_offset,
            -0.5 * h + t + y_offset,
            0.5 * h,
        ]
        self.eps_lists = [
            [eps_uc, eps_lc, eps_lc, eps_core, eps_uc],
            [eps_uc, eps_lc, eps_core, eps_core, eps_core],
        ]

    def ridge_pn(self, eps0, epsp, epsn, eps_lc, eps_uc, w, h, t, xp, xn):
        if xp < -0.5 * w:
            x_left = [xp, -0.5 * w]
            eps_left = [[eps_uc, eps_lc, epsp, eps_uc], [eps_uc, eps_lc, eps0, eps_uc]]
        else:
            x_left = [-0.5 * w, xp]
            eps_left = [[eps_uc, eps_lc, epsp, eps_uc], [eps_uc, eps_lc, epsp, epsp]]
        if xn > 0.5 * w:
            x_right = [0.5 * w, xn, 0.5]
            eps_right = [
                [eps_uc, eps_lc, eps0, eps0],
                [eps_uc, eps_lc, eps0, eps_uc],
                [eps_uc, eps_lc, epsn, eps_uc],
            ]
        else:
            x_right = [xn, 0.5 * w, xn]
            eps_right = [
                [eps_uc, eps_lc, eps0, eps0],
                [eps_uc, eps_lc, epsn, epsn],
                [eps_uc, eps_lc, epsn, eps_uc],
            ]

        self.x_list = x_left + x_right
        self.y_list = [-0.5, -0.5 * h, -0.5 * h + t, 0.5 * h]
        self.eps_lists = eps_left + eps_right

    def ridge_double(
        self, eps_core, eps_lc, eps_uc, w1, w2, h, t1, t2, y_offset=0.0, x_offset=0.0
    ):
        """Rib waveguide with double etch

        Args:
            eps_core (float):   epsilon of the core
            eps_lc (floar):     epsilon of the lower cladding
            eps_up (float):     epsilon of the upper cladding
            w1 (float):         width of the unetched part (in unit of ax)
            w2 (float):         width of the intermediate etched part (in unit of ax)
            h (float):          height of the un-etched part (in unit of ay)
            t1 (float):         height of the intermidiate etched part (in unit of ay).
            t2 (float):         height of the maximum etched part (in unit of ay).
            x_offset (float):   offset of the center of the waveguide with respec to the center of the cell (in unit of ax). Default is 0
            y_offset (float):   offset of the etched part with resoect to the unetched one (in unit of ay). Default is 0 (etched and unetched part are aligned at the bottom)

        """

        self.x_list = [
            -0.5 * w2 + x_offset,
            -0.5 * w1 + x_offset,
            0.5 * w1 + x_offset,
            0.5 * w2 + x_offset,
        ]
        self.y_list = [
            -0.5,
            -0.5 * h,
            -0.5 * h + y_offset,
            -0.5 * h + t2 + y_offset,
            -0.5 * h + t1 + y_offset,
            0.5 * h,
        ]
        self.eps_lists = [
            [eps_uc, eps_lc, eps_lc, eps_core, eps_uc, eps_uc],
            [eps_uc, eps_lc, eps_lc, eps_core, eps_core, eps_uc],
            [eps_uc, eps_lc, eps_core, eps_core, eps_core, eps_core],
            [eps_uc, eps_lc, eps_lc, eps_core, eps_core, eps_uc],
        ]

    def rect(self, eps_core, eps_clad, w, h, off_x=0.0, off_y=0.0):
        """Rectangular waveguide

        Args:
            eps_core (float):   epsilon of the core
            eps_clad (floar):   epsilon of the cladding
            w (float):          width of the waveguide (in unit of ax)
            h (float):          height of the waveguide (in unit of ay)
            off_y (float):      offset of the center of the waveguide with respect to the cell (in unit of ay). Default is 0.
            off_x (float):      offset of the center of the waveguide with respect to the cell (in unit of ax). Default is 0.
        """
        self.x_list = [-0.5 * w + off_x, 0.5 * w + off_x]
        self.y_list = [-0.5 * h + off_y, 0.5 * h + off_y]
        self.eps_lists = [[eps_clad, eps_clad], [eps_clad, eps_core]]

    def slab(self, eps_core, eps_lc, eps_uc, w, offset=0.0):
        """1D slab in x direction

        Args:
            eps_core (float): epsilon of the core.
            eps_lc (float): epsilon of the lower cladding.
            eps_uc (float): epsilon of the upper cladding.
            w (float): thickness of the slab (in unit of ax).
            offset (float, optional): Offset if the slab with respect to the center of the cell. Defaults to 0.0.

        Returns:
            None.

        """
        self.x_list = [-0.5, -0.5 * w + offset, 0.5 * w + offset]
        self.y_list = [0.5]
        self.eps_lists = [[eps_uc], [eps_lc], [eps_core]]

    def slab_y(self, eps_core, eps_lc, eps_uc, w):
        """1D slab in y direction

        Args:
            eps_core (float): epsilon of the core.
            eps_lc (float): epsilon of the lower cladding.
            eps_uc (float): epsilon of the upper cladding.
            w (float): thickness of the slab (in unit of ay).
            offset (float, optional): Offset if the slab with respect to the center of the cell. Defaults to 0.0.

        Returns:
            None.

        """
        self.x_list = [0.5]
        self.y_list = [-0.5, -0.5 * w, 0.5 * w]
        self.eps_lists = [[eps_uc, eps_lc, eps_core]]

    def x_stack(self, x_l, eps_l):
        self.y_list = [0.5]
        self.x_list = [-0.5] + x_l
        self.eps_lists = [[eps_l[-1]]]
        for eps in eps_l:
            self.eps_lists.append([eps])

    def hole(self, h, w, r, e_core, e_lc, e_up, e_fill):
        """Rib waveguide with a hole in the middle


        Args:
            h (TYPE): height of the waveguide (in unit of ay).
            w (TYPE): width of the waveguide (in unit of ax).
            r (TYPE): radius of the internal hole (in unit of ax).
            e_core (TYPE): epsilon of the core.
            e_lc (TYPE): epsilon of the lower cladding.
            e_up (TYPE): epsilon of the upper cladding.
            e_fill (TYPE): epsilon inside the hole.

        Returns:
            None.

        """
        self.x_list = [-0.5 * w, -r, r, 0.5 * w]
        self.y_list = [-0.5 * h, 0.5 * h, 0.5]
        self.eps_lists = [
            [e_lc, e_up, e_up],
            [e_lc, e_core, e_up],
            [e_lc, e_fill, e_up],
            [e_lc, e_core, e_up],
        ]

    def circle(self, e_in, e_out, r, n):
        self.x_list = np.linspace(-r, r, n)
        self.y_list = np.linspace(-r, r, n)
        [X, Y] = np.meshgrid(self.x_list, self.y_list)
        # ind= np.sqrt((X-0.5*r/float(n))**2+(Y-0.5*r/float(n))**2)<r
        ind = np.sqrt(X**2 + Y**2) < r
        # eps=np.array([e_out,e_in])
        self.eps_lists = e_out + ind * (e_in - e_out)

    def etched_stack(self, eps_uc, eps_lc, w, etch, eps_stack, d_stack):
        h = sum(d_stack)
        self.x_list = [-0.5 * w, 0.5 * w]
        self.y_list = [-0.5]
        eps1 = [eps_uc, eps_lc]
        eps2 = [eps_uc, eps_lc]
        dd = np.cumsum(d_stack)
        if etch > h:
            self.y_list.append(0.5 * h - etch)
            eps1.append(eps1[-1])
            eps2.append(eps_uc)
            dec = 1
        else:
            dec = 0
        for d, eps in zip(reversed(dd), reversed(eps_stack)):
            if (d < etch) and (dec == 0):
                self.y_list.append(0.5 * h - etch)
                eps1.append(eps1[-1])
                eps2.append(eps_uc)
                dec = 1
            self.y_list.append(0.5 * h - d)
            eps1.append(eps)
            if dec == 0:
                eps2.append(eps)
            else:
                eps2.append(eps_uc)
        self.y_list.append(0.5 * h)
        self.eps_lists = [eps2, eps1]

    def varied_epi(self, eps_back, data_list, y_off=0.0):
        t_tot = sum([dd[2] for dd in data_list])
        w_list = np.sort(list(set([0.5 * dd[1] for dd in data_list])))
        self.x_list = [-0.5] + list(-w_list[::-1]) + list(w_list)
        self.y_list = [-0.5 * t_tot] + list(
            -0.5 * t_tot + np.cumsum([dd[2] for dd in data_list])
        )
        self.y_list = [_ + y_off for _ in self.y_list]
        self.eps_lists = [len(self.y_list) * [eps_back]]
        for pos in self.x_list:
            eps_list = [eps_back]
            for eps, w, t in data_list:
                if pos < -0.5 * w:
                    eps_list.append(eps_back)
                elif pos >= 0.5 * w:
                    eps_list.append(eps_back)
                else:
                    eps_list.append(eps)
            self.eps_lists.append(eps_list)

    def varied_plane(self, eps_back, t, data_list):
        self.y_list = [-0.5 * t, 0.5 * t]
        w_tot = sum([dd[1] for dd in data_list])
        self.x_list = [-0.5 * w_tot] + list(
            -0.5 * w_tot + np.cumsum([dd[1] for dd in data_list])
        )
        self.eps_lists = [len(self.y_list) * [eps_back]]
        for i, (eps, w) in enumerate(data_list):
            eps_list = [eps_back, eps]
            self.eps_lists.append(eps_list)

        # print(self.y_list)
        # print(self.x_list)
        # print(self.eps_lists)

    def plot_eps(self, N=101):
        EPS = np.zeros((N, N)) + self.eps_lists[0][0]
        x = np.linspace(-0.5, 0.5, N)
        y = np.linspace(-0.5, 0.5, N)
        x, y = np.meshgrid(x, y, indexing="ij")
        x_old = -0.5
        for xv, el in zip(self.x_list, self.eps_lists):
            EPS[np.logical_and(x >= xv, x <= x_old)] = el[0]
            for yv, e in zip(self.y_list, el):
                EPS[np.logical_and(x >= xv, y >= yv)] = e
                plt.imshow(np.logical_and(x <= xv, x >= x_old, y >= yv))
                plt.show()
                x_old = xv
        return EPS
