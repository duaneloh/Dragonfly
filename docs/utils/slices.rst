Tomographic Slices
=================

Generate slices from 3D intensity distributions for given orientations.

.. automodule:: dragonfly.utils.py_src.slices
   :members:
   :undoc-members:
   :show-inheritance:

Example
-------

.. code-block:: python

   from dragonfly.utils.py_src.slices import SliceGenerator

   gen = SliceGenerator('config.ini')
   tomo, info = gen.get_slice(10, 5)
