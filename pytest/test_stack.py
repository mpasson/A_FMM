import aptdaemon.console
import numpy as np
import pickle

from A_FMM import Creator, Layer, Layer_uniform, Stack

def test_1D_coefficients():
    with open('pickles/test_stack_coefficients.pkl', 'rb') as f:
        coeff_ref = pickle.load(f)

    lay1 = Layer_uniform(0,0,2.0)
    lay2 = Layer_uniform(0,0,12.0)
    stack = Stack(
        10 * [lay1, lay2] + [lay1],
        10*[0.5, 0.5]   + [0.5],
    )
    stack.solve(0.1)

    for (u, d, lay, t), (uref, dref) in zip(stack.loop_intermediate([1.0, 0.0], [0.0, 0.0]), coeff_ref):
        assert np.allclose(u, uref)
        assert np.allclose(d, dref)

def test_1D_field():
    with open('pickles/test_stack_1Dfield.pkl', 'rb') as f:
        field_ref = pickle.load(f)

    lay1 = Layer_uniform(0,0,2.0)
    lay2 = Layer_uniform(0,0,12.0)
    stack = Stack(
        10 * [lay1, lay2] + [lay1],
        [0.0] + 10*[0.5, 0.5],
    )
    x, y, z = 0.0, 0.0, np.linspace(0.0, 10.0, 1000)
    stack.solve(0.1)
    field = stack.calculate_fields([1.0, 0.0], [0.0, 0.0], x, y, z)
    for key, value in field.items():
        assert np.allclose(value, field_ref[key])


def test_hugonin():
    ax = 1.0
    lam = 0.975
    k0 = ax / lam

    s = 0.3
    d = 0.15

    n_core = 3.5
    n_clad = 2.9
    n_air = 1.0

    Nx = 10
    Ny = 0

    cr = Creator()
    cr.slab(n_core ** 2.0, n_clad ** 2.0, n_air ** 2.0, s / ax)
    wave = Layer(Nx, 0, cr)
    cr.slab(n_air ** 2.0, n_clad ** 2.0, n_air ** 2.0, s / ax)
    gap = Layer(Nx, 0, cr)
    mat = [wave, gap, wave, gap, wave]
    dl = [x/ax for x in [1.0,d,d,d,1.0]]
    st = Stack(mat, dl)
    #st.transform_complex(0.7)
    st.transform(ex=0.7, complex_transform=True)
    st.solve(ax/lam)
    assert np.allclose(st.get_R(0,0), 0.38787321)
    assert np.allclose(st.get_R(1, 1), 0.36478382)