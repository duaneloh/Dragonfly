Data Stream Simulator
=====================

The data stream simulator generates synthetic single-particle diffraction data
from PDB files, including realistic noise and photon statistics.

sim_setup
---------

Convenience utility that runs the simulator pipeline sequentially::

    dragonfly.utils.sim_setup

make_densities
--------------

Creates electron density map from PDB file.

**Usage**::

    dragonfly.utils.make_densities
    dragonfly.utils.make_densities -c config.ini

**Output**: Binary file containing 3D cubic electron density distribution
at the appropriate voxel size.

A low-pass filter eliminates artifacts from electron density
discretization on a cubic grid.

make_intensities
----------------

Creates 3D intensity map from electron density with low-pass filtering.

**Usage**::

    dragonfly.utils.make_intensities
    dragonfly.utils.make_intensities -c config.ini

**Output**: Binary file containing 3D intensity distribution in reciprocal space.

make_detector
-------------

Generates detector geometry file with pixel information.

**Usage**::

    dragonfly.utils.make_detector
    dragonfly.utils.make_detector -c config.ini

**Output**: Detector geometry file (``.h5`` or ``.dat``) containing:

* Detector distance
* Pixel size
* Beam center coordinates
* Mask information

make_data
---------

Simulates photon diffraction patterns from 3D intensity volume.

**Usage**::

    dragonfly.utils.make_data
    dragonfly.utils.make_data -c config.ini

**Output**: Binary file (``.emc``) containing sparse photon data.

Data Format
-----------

The simulator outputs photon data in sparse format with three arrays:

#. **Single-photon locations** (variable):
   - Pixel indices where exactly one photon was detected

#. **Multi-photon locations** (variable):
   - Pixel indices for multi-photon events

#. **Multi-photon counts** (variable):
   - Photon count at each multi-photon pixel

This sparse format is efficient for high-resolution SPI where most
pixels receive zero or one photon.

Pipeline Flow
-------------

::

    PDB file
       |
       v
    make_densities --> densityMap.bin
       |
       v
    make_intensities --> intensities.bin
       |
       v
    make_detector --> det_geom.dat
       |
       v
    make_data --> photons.emc
