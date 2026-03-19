Welcome to Dragonfly's documentation!
=====================================

Dragonfly is a software package for single-particle diffractive imaging data analysis,
implementing the EMC (Expand-Maximize-Compress) single-particle reconstruction algorithm
using MPI and OpenMP.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   detector
   emcfile
   model
   quaternion
   iterate
   params
   recon
   utils

Overview
--------

The package consists of several Cython modules that provide an interface to
optimized C code for high-performance reconstruction:

* ``CDetector`` / ``Detector`` - Detector geometry and pixel information
* ``CDataset`` / ``EMCReader`` / ``EMCWriter`` - EMC data file I/O
* ``Model`` - 3D/2D volume model management
* ``Quaternion`` - Orientation sampling for SO(3)
* ``Iterate`` - Iteration state management
* ``EMCParams`` - Reconstruction parameters
* ``EMCRecon`` - Main reconstruction class

Quick Start
-----------

1. Install the package::

    pip install -e .

2. Initialize a new reconstruction::

    dragonfly.init -c config.ini

3. Run reconstruction::

    mpirun -n 4 dragonfly.emc 100 -c config.ini


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
