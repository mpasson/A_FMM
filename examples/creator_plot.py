#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 22:24:15 2021

@author: marco
"""

import A_FMM
import matplotlib.pyplot as plt

cr = A_FMM.creator()

#cr.slab(12.0, 2.0, 2.0, 0.2)
#cr.circle(12.0,2.0,0.3,21)
cr.ridge(12.0, 2.1, 2.1, 0.4, 0.3, t=0.15)
eps = cr.plot_eps()
plt.imshow(eps)
plt.show()