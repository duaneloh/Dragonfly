Frequently Asked Questions
==========================

Installation
------------

**How do I install dragonfly on an HPC cluster?**

Load your MPI module first, then install::

    module load mpi/openmpi-4.1
    pip install dragonfly-spi

**Why can't pip find mpi4py?**

 mpi4py must be installed after loading your system's MPI library.
 It will not work if installed before the MPI module is loaded.

**Can I use conda instead of pip?**

Yes, but ensure mpi4py is installed via conda with the correct MPI::

    conda install mpi4py openmpi

Data Processing
---------------

**What file formats does dragonfly support?**

* Binary photon files (``.emc``)
* Detector files (``.h5``)
* PDB files for simulations

**How do I convert my data to dragonfly format?**

Use the conversion utilities in ``utils/convert/``::

    python utils/convert/cxi_to_photons.py -i data.cxi -o data.photons

**How do I choose the orientation sampling density (num_div)?**

This is a function of your dimensionless resolution, counted as the
ratio of the size of your particle to your resolution, which is also 
around twice the highest order speckle/fringe.

The total number of orientations is ``10*(5*num_div^3 + num_div)``:

========  ============
num_div   Orientations
========  ============
2         420
4         3240
6         10860
8         25680
========  ============

The average angle between neighboring orientations is around ``0.91/num_div``
radians. A rule of thumb is to use ``num_div`` sampling for reconsutrction up 
to a radius of ``num_div`` speckles/fringes.

Reconstruction
--------------

**How do I resume an interrupted reconstruction?**

Use the ``-r`` flag::

    dragonfly.emc -c config.ini -r 20

This resumes for 20 more iterations from the last checkpoint.

**What do the output volumes represent?**

Volumes contain the 3D intensity distribution of the sample in
reciprocal space. The voxel size can be calculated as::

    1 / (lambda * ewald_rad)

where ``ewald_rad`` is the Ewald sphere radius in voxels, specified
in the detector file.

**The GUI doesn't display properly.**

The autoplot GUI requires a display. On remote systems:

* Use X11 forwarding: ``ssh -X``

You can check the ``$DISPLAY`` environment variable to check.

**How do I visualize the reconstructed volumes?**

Dragonfly outputs h5 files for every iteration. Use:

* The autoplot GUI
* Custom python code to read the ``intens/`` dataset.

Performance
-----------

**How many cores should I use?**

Optimal configuration depends on your system:

* Total cores = MPI processes × OpenMP threads

For a 64-core node::

    mpirun -n 8 -c 8 dragonfly.emc   # 8 processes, 8 threads each

**Why are the earlier iterations slower?**

The first iterations typically have broader probability distributions,
leading to slower updated tomogram calculations. At the end, frames are
often assigned to single orientations, which makes the ``update`` step
faster.

**Can I run on GPUs?**

Currently, dragonfly uses CPU-only computation. GPU acceleration
is planned for a future release called `DragonflyX`.

Getting Help
------------

For additional support:

* GitHub Issues: https://github.com/duaneloh/Dragonfly/issues
* Documentation: https://dragonfly-spi.readthedocs.io
