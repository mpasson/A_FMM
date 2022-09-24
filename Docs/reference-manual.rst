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
    Layer_num
    Layer_uniform

Methods
-------------

.. autosummary::
    :toctree: generated/

    Layer._check_array_shapes
    Layer._filter_componets
    Layer._process_xy
    Layer.add_transform_matrix
    Layer.calculate_epsilon
    Layer.calculate_field
    Layer.calculate_field_old
    Layer.clear
    Layer.coupling
    Layer.create_input
    Layer.eps_plot
    Layer.get_Enorm
    Layer.get_P_norm
    Layer.get_Poyinting_norm
    Layer.get_Poynting
    Layer.get_Poynting_single
    Layer.get_index
    Layer.get_input
    Layer.get_modal_field
    Layer.inspect
    Layer.interface
    Layer.mat_plot
    Layer.mode
    Layer.overlap
    Layer.plot_Ham
    Layer.transform



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
    Stack.bloch_modes
    Stack.calculate_epsilon
    Stack.calculate_fields
    Stack.count_interface
    Stack.double
    Stack.flip
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
    Stack.loop_intermediate
    Stack.solve
    Stack.solve_S
    Stack.solve_lay
    Stack.solve_serial
    Stack.transform



S_matrix
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




