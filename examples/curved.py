import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import A_FMM 


Nx,Ny = 10,10

ax=4.0
ay=2.0
r=2.0

cr = A_FMM.creator()
cr.rect(12.0,2.0,0.5/ax,0.2/ay)

wave_s = A_FMM.layer(Nx,Ny,cr, Nyx=ay/ax)
wave_s.transform_complex(ex=0.8, ey=0.8)
wave_s.mode(ax/1.55)
N=wave_s.get_index()[0]
print('inf', N)

for radius in np.linspace(1.0, 10.0, 10):
    wave_bend = A_FMM.layer_curved.from_layer(wave_s, radius/ax)
    #wave_bend.transform(ex=0.8, ey=0.8)
    wave_bend.transform_complex(ex=0.8, ey=0.8)
    wave_bend.mode(ax/1.55)
    N=wave_bend.get_index()[0]
    print(radius, N)


