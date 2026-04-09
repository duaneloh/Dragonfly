Common-line CC Calculator
=========================

Calculate correlation coefficients between model slices.

.. automodule:: dragonfly.utils.py_src.calc_cc
   :members:
   :undoc-members:
   :show-inheritance:

CCCalculator Class
------------------

.. autoclass:: dragonfly.utils.py_src.calc_cc.CCCalculator
   :members:
   :undoc-members:
   :noindex:

Example
-------

.. code-block:: python

   import h5py
   from dragonfly.utils.py_src.calc_cc import CCCalculator

   with h5py.File('output_010.h5', 'r') as f:
       intens = f['intens'][:]

   calc = CCCalculator(intens, n_angbins=40, mask_radius=20)
   cc_matrix = calc.run(nproc=16)
