import numpy as np
import matplotlib.pyplot as plt

from A_FMM import Layer, Creator

cr = Creator()
cr.rect(12.0, 2.0, 0.6, 0.3)
lay = Layer(10, 10, cr)
lay.transform(ex=0.8, ey=0.8)
lay.mode(2.0)
field = lay.get_modal_field(0, np.linspace(-0.5, 0.5, 101), np.linspace(-0.5, 0.5, 101))
print(field.keys())
plt.contourf(field['x'], field['y'], field['Ex'])
plt.show()