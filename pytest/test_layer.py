import numpy as np
import pickle

from A_FMM import Creator, Layer

def create_later_from_creator():
    cr = Creator()
    cr.rect(12.0, 2.0, 0.6, 0.3)
    lay = Layer(5, 5, cr)
    return lay

def test_layer_creation():
    with open('test_layer_creation.pkl', 'rb') as pkl:
        lay_ref = pickle.load(pkl)
    lay = create_later_from_creator()
    for attr in ['FOUP', 'INV', 'EPS1', 'EPS2']:
        assert np.allclose(getattr(lay, attr), getattr(lay_ref, attr))

def test_layer_transform():
    with open('test_layer_transform.pkl', 'rb') as pkl:
        lay_ref = pickle.load(pkl)
    lay = create_later_from_creator()
    lay.transform(ex=0.8, ey=0.8)
    for attr in ['FOUP', 'INV', 'EPS1', 'EPS2', 'FX', 'FY']:
        assert np.allclose(getattr(lay, attr), getattr(lay_ref, attr))


if __name__ == '__main__':
    lay = create_later_from_creator()
    with open('test_layer_creation.pkl', 'wb') as pkl:
        pickle.dump(lay, pkl, protocol=pickle.HIGHEST_PROTOCOL)
    lay.transform(ex=0.8, ey=0.8)
    with open('test_layer_transform.pkl', 'wb') as pkl:
        pickle.dump(lay, pkl, protocol=pickle.HIGHEST_PROTOCOL)
