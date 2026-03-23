Utilities
==========

The utilities directory contains Python utilities, command-line tools, and Cython
extensions for data processing and simulation.

Command-line Tools
------------------

The package provides several command-line tools:

* ``dragonfly.init`` - Initialize new reconstruction
* ``dragonfly.emc`` - Run EMC reconstruction
* ``dragonfly.autoplot`` - Automatic plotting of results
* ``dragonfly.frameviewer`` - Interactive frame viewer
* ``dragonfly.make_densities`` - Generate density maps
* ``dragonfly.make_intensities`` - Generate intensity maps
* ``dragonfly.make_detector`` - Create detector files
* ``dragonfly.sim_setup`` - Setup simulations
* ``dragonfly.make_data`` - Generate simulated data
* ``dragonfly.compare`` - Compare reconstructions

Simulation Utilities
--------------------

.. automodule:: dragonfly.utils.make_densities
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: dragonfly.utils.make_intensities
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: dragonfly.utils.make_detector
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: dragonfly.utils.sim_setup
   :members:
   :undoc-members:
   :show-inheritance:

Configuration Parser
--------------------

.. automodule:: dragonfly.utils.py_src.read_config
   :members:
   :undoc-members:
   :show-inheritance:

.. toctree::
   :maxdepth: 1

   utils/py_utils
   utils/slices
   utils/calc_cc