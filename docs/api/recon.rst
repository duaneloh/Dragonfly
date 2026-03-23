Reconstruction Module
=====================

The reconstruction module performs the main EMC reconstruction.

Usage
-----

The reconstruction is typically run via the command line:

.. code-block:: bash

   mpirun -n 4 dragonfly.emc -c config.ini -t 8 100

Or programmatically:

.. code-block:: python

   from dragonfly.recon import EMCRecon
   from dragonfly.iterate import Iterate

   recon = EMCRecon(num_threads=8)
   itr = Iterate('config.ini')
   itr.params.num_iter = 100
   recon.set_iterate(itr)

   for itr.params.iteration in range(1, itr.params.num_iter+1):
       recon.run_iteration()

In each case, 100 iterations are run from a random start using 4 MPI ranks and 8 threads per rank.

Output Files
------------

Each iteration produces an ``output_XXX.h5`` HDF5 file containing:

* ``intens`` - Current model intensities
* ``orientations`` - Most likely orientations
* ``mutual_info`` - Mutual information per frame
* ``likelihood`` - Log-likelihood per frame
* ``quaternions`` - Orientation quaternions
* ``params/*`` - Reconstruction parameters

.. automodule:: dragonfly.recon
   :members:
   :undoc-members:
   :show-inheritance:
