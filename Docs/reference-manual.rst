.. currentmodule:: A_FMM

*************
API Reference
*************


Creator
+++++++++++++++


.. autosummary::
    :toctree: generated/

    Creator

Methods
-------------

.. autosummary::
    :toctree: generated/

    Creator.circle
    Creator.etched_stack
    Creator.hole
    Creator.plot_eps
    Creator.rect
    Creator.ridge
    Creator.ridge_double
    Creator.ridge_pn
    Creator.slab
    Creator.slab_y
    Creator.slow_2D
    Creator.slow_general
    Creator.varied_epi
    Creator.varied_plane
    Creator.x_stack


Layer
+++++++++++++++


.. autosummary::
    :toctree: generated/

    Layer
    Layer_ani_diag
    Layer_empty_st
    Layer_from_hstack
    Layer_from_xsection
    Layer_num
    Layer_uniform

Methods
-------------

.. autosummary::
    :toctree: generated/

    Layer.add_transform_matrix
    Layer.clear
    Layer.coupling
    Layer.create_input
    Layer.eps_plot
    Layer.get_Enorm
    Layer.get_P_norm
    Layer.get_Poyinting_norm
    Layer.get_Poynting
    Layer.get_Poynting_single
    Layer.get_field
    Layer.get_field2
    Layer.get_index
    Layer.get_input
    Layer.inspect
    Layer.interface
    Layer.mat_plot
    Layer.mode
    Layer.overlap
    Layer.plot_E
    Layer.plot_Et
    Layer.plot_H
    Layer.plot_Ham
    Layer.plot_field
    Layer.slim
    Layer.transform
    Layer.transform_complex
    Layer.writeE
    Layer.writeH
    Layer.write_field
    Layer.write_fieldgeneral


Stack
+++++++++++++++


.. autosummary::
    :toctree: generated/

    Stack


Methods
-------------

.. autosummary::
    :toctree: generated/


    Stack.add_layer
    Stack.add_transform
    Stack.add_transform_complex
    Stack.bloch_modes
    Stack.count_interface
    Stack.create_input
    Stack.double
    Stack.flip
    Stack.fourier
    Stack.get_PR
    Stack.get_PT
    Stack.get_R
    Stack.get_T
    Stack.get_el
    Stack.get_energybalance
    Stack.get_inout
    Stack.get_prop
    Stack.inspect
    Stack.join
    Stack.line_E
    Stack.mat_plot
    Stack.mode_T
    Stack.plot_E
    Stack.plot_EY
    Stack.plot_E_general
    Stack.plot_E_periodic
    Stack.plot_E_plane
    Stack.plot_Ex
    Stack.plot_Ey
    Stack.plot_stack
    Stack.plot_stack_y
    Stack.solve
    Stack.solve_S
    Stack.solve_lay
    Stack.solve_serial
    Stack.transform
    Stack.transform_complex
    Stack.writeE
    Stack.writeE_periodic_XY
    Stack.writeE_periodic_XZ
    Stack.writeE_periodic_YZ


Stack
+++++++++++++++


.. autosummary::
    :toctree: generated/

    S_matrix


Methods
-------------

.. autosummary::
    :toctree: generated/

    S_matrix.add
    S_matrix.add_left
    S_matrix.add_uniform
    S_matrix.add_uniform_left
    S_matrix.der
    S_matrix.det
    S_matrix.det_modes
    S_matrix.int_f
    S_matrix.int_f_tot
    S_matrix.left
    S_matrix.matrix
    S_matrix.output




