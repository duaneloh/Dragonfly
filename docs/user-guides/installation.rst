Installation
============

Dragonfly can be installed from PyPI or built from source. The package
requires MPI for parallel execution.

Prerequisites
-------------

System dependencies:

* **MPI** (OpenMPI or MPICH)
* **GSL** (GNU Scientific Library)
* **HDF5** (with parallel support)
* **OpenMP** (usually included with compiler)

Python dependencies:

* Python 3.8+
* numpy
* scipy
* h5py
* mpi4py
* cython

Install from PyPI
-----------------

The easiest way to install is via pip::

    pip install dragonfly-spi

For HPC environments with MPI modules::

    module load mpi/openmpi-x.x.x
    pip install mpi4py
    pip install dragonfly-spi

.. note::

    mpi4py must be installed after loading your system's MPI library.
    It will not work if installed before the MPI module is loaded.

Install from Source
-------------------

Clone the repository::

    git clone https://github.com/duaneloh/Dragonfly.git
    cd Dragonfly

Install Python dependencies::

    pip install -r requirements.txt

.. note::

    On HPC systems, load your MPI module before installing::

        module load mpi/openmpi-x.x.x
        pip install mpi4py
        pip install -r requirements.txt

Install the package with Cython extensions::

    pip install -e .

This builds the Cython extensions in-place. For a fresh build::

    pip install -e . --no-build-isolation

Build the C Executable
----------------------

For the full EMC reconstruction, build the C executable with CMake::

    mkdir -p build && cd build
    cmake .. && make

This requires:

* MPI development headers
* GSL development libraries
* HDF5 with parallel support
* OpenMP support

Verify Installation
-------------------

To verify the Python package is installed correctly::

    python -c "import dragonfly; print(dragonfly.__version__)"

To check MPI is working::

    mpirun --version

Uninstall
---------

To uninstall the package::

    pip uninstall dragonfly-spi

or from source::

    pip uninstall dragonfly

Troubleshooting
---------------

**mpi4py installation fails**

    Ensure your MPI module is loaded before installing mpi4py::

        module load mpi/openmpi-x.x.x
        pip install mpi4py

**Import errors after installation**

    Try reinstalling with a fresh build::

        pip uninstall dragonfly
        pip install -e . --no-build-isolation

**CMake cannot find HDF5**

    Install HDF5 with parallel support or set HDF5_DIR environment variable.

**Missing GSL library**

    Install GSL development libraries (e.g., ``libgsl-dev`` on Debian/Ubuntu).

Next Steps
----------

* :doc:`simulation-quickstart` - Run your first simulation
* :doc:`experimental-quickstart` - Process publicly available experimental data
