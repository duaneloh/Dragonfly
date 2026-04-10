Quick Start with Simulations
============================

This guide walks through the simulation workflow for single-particle imaging.

.. image:: /_static/images/emc_sim.png
   :alt: Simulation pipeline flowchart
   :width: 60%

For installation instructions, see :doc:`installation`.

Spawn a Reconstruction Directory
--------------------------------

Go to a directory with plenty of space and start a reconstruction session::

    dragonfly.init

This creates a new reconstruction directory named ``recon_0001`` with all necessary
binaries and configuration files.

For customization options::

    dragonfly.init -h

Configure Experiment
--------------------

Go to your newly created reconstruction directory::

    cd recon_0001

Edit ``config.ini`` to customize parameters:

* ``in_pdb_file``: Path to your PDB file
* Detector setup (in ``[parameters]`` section):

  * ``detd``: Detector distance (mm)
  * ``lambda``: Photon wavelength (Angstrom)
  * ``detsize``: Linear detector size (pixels)
  * ``pixsize``: Pixel size (mm)
  * ``stoprad``: Beamstop radius (pixels)

* ``num_data``: Number of diffraction patterns
* ``fluence``: Incident beam intensity (photons/um^2)
* ``log_file``: Name of log file

Generate simulated data::

    dragonfly.utils.sim_setup

More details in :doc:`simulator` for a description
of the constituent utilities.

Run EMC Reconstruction
----------------------

**On a single node (only OpenMP)**::

    dragonfly.emc -t <num_threads> 10

Starts reconstruction with default ``config.ini`` config file for 10 iterations
using `<num_threads>` threads.

**Continue reconstruction with maximum threads**::

    dragonfly.emc -r 20

**MPI reconstruction using different config file**::

    mpirun -np 4 dragonfly.emc -c config2d.ini 50

Often these larger reconstructions are submitted through a job scheduler like
SLURM, PBS etc.

Monitor Progress
----------------

In the reconstruction directory::

    dragonfly.autoplot -c <config_file>

This GUI monitors reconstruction progress and can automatically look for new
reconstructed volumes. See :doc:`autoplot` for full documentation.

.. image:: /_static/images/Screenshot_for_quickstart.png
   :alt: Progress Viewer after simulation reconstruction
   :width: 100%

Example Workflow
----------------

Complete example with 8 cores::

    dragonfly.init                     # Spawn new reconstruction directory
    cd recon_0001                      # Change into new directory
    ./utils/sim_setup.py               # Setup data stream simulator
    mpirun -n 2 dragonfly-emc 30 -t 4  # Run MPI+OpenMP reconstruction
    dragonfly.autoplot                 # Monitor progress

Additional Resources
--------------------

* See ``sample_configs/`` for example configurations
* Check ``logs/EMC.log`` and ``recon.log`` for detailed logs
* Refer to :doc:`config` for description of configuration parameters
* Refer to :doc:`simulator` for data stream simulator details
