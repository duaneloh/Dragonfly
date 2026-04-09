Model Module
============

The model module manages 3D/2D volume models for reconstruction.

Model Types
-----------

* ``MODEL_3D`` - Full 3D volume reconstruction
* ``MODEL_2D`` - 2D slice reconstruction
* ``MODEL_RZ`` - Cylindrical (R-Z) reconstruction

Usage Example
-------------

.. code-block:: python

   from dragonfly.model import Model

   model = Model(size=64, num_modes=2, model_type='3d')
   model.allocate('starting_model.h5', model_mean=100.0)

   view = model.slice_gen(quaternion, detector, mode=0)

.. automodule:: dragonfly.model
   :members:
   :undoc-members:
   :show-inheritance:
