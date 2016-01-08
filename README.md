# EMC
--------------------------------------------------------------------------------
## Quick start  
### Spawn a reconstruction directory
Type in the topmost directory:
	./setup_new_recon.py
This makes the binaries needed for the package and links them into a new reconstruction directory named "recon_xxxx". For help on customizing this setup script (e.g. custom folder name or counter):
	./setup_new_recon.py -h

### Configure experiment
Go to your newly created recon directory
	cd recon_xxxx
Change the experiment parameters in config.ini to your liking. 
Here are some things that you might like to change:
- [in_pdb_file] relative path to your own PDB file
- scattering setup (detector distance, photon wavelength, etc)
- [num_data] the number of diffraction patterns
- [mean_count] the average number of photons per pattern
- [log_file] name of log file

When ready to start creating synthetic data, type:
	./sim_setup.py
Again, you can get help to customize this using the command:
	./sim_setup.py -h

### Start your EMC reconstruction
You can start a single process reconstruction in the recon directory this way:
	./emc <num_iterations> <path to config file> [threads per process]
where <necessary arguments> and [optional arguments]. The default number of threads per process is defined by system parameter OMP_NUM_THREADS.

To spawn multiple MPI reconstructions from your recon directory:
	mpirun -n <num_mpi_processes> ./emc <num_iterations> <path to config file> [threads per process]

### Monitor the progress of your reconstruction 
In the recon directory type:
	./autoplot.py

--------------------------------------------------------------------------------
## exp_config.dat
Structure of exp_config.dat

--------------------------------------------------------------------------------
## make_densities.py
Create electron density map from PDB file

Densities.dat is a flattened ASCII record of the floating point numbers of a 3D cubic electron density distribution. 

### Usage and parameters
To get usage help:
	python make_densities.py -h
Make densities from experimental parameters specified in the config file "exp_config.dat"
	python make_densities.py <exp_config.dat>

--------------------------------------------------------------------------------
## make_intensities.py 
Create 3D intensity map from electron density
### Usage and parameters
To get usage help:
	python make_intensities.py -h
Make intensities from experimental parameters specified in the config file "exp_config.dat"
	python make_intensities.py <exp_config.dat>

--------------------------------------------------------------------------------
## make_detector.py
This module generates the detector.dat file which contains information about
each detector pixel. 
### Input
exp_config.dat
### Output
detector.dat

The detector.dat file is an ASCII (human-readable) file.
 - The first line contains a single integer which is the number of pixels
 - From the second line on, there are 5 columns per line. Each line represents
   information about a pixel. The pixel index in the rest of this package
   refers to the line number in this file.
 - The first three columns give the 3D reciprocal space coordinates of the pixel
   relative to the 3D intensity model.
 - The fourth column is a multiplicative factor for solid angle and polarization (x or y direction) corrections for the pixel.
 - The fifth column is a mask value. There are three types of pixels:
 	- 0: Good pixels. Assumed to have accurate photon information and will be
	  used to determine the orientation.
	- 1: Non-orientationally relevant pixels. They have accurate photon
	  information but are not used to determine the orientation. These are
	  generally pixels at the corners of the detector, which sample regions of
	  reciprocal space with very low multiplicity. They are excluded to avoid
	  overfitting of orientations.
	- 2: Bad pixels. These pixels are completely ignored. In principle, they can
	  be removed from the detector.dat file completely. This mechanism is in
	  place if one wants space-filling detector shapes for convenience.

### Usage and parameters
To use this module, just run `make` and the `./utils/make_detector`. All the
parameters will be taken from `config.ini`. An example file has been provided
with the repository. Copy it over using `cp config.ini.example config.ini` and
edit the experimental parameters. The name of the output file is set by the
variable 'detector'.


--------------------------------------------------------------------------------
## make_quaternion.c
### Input
None
### Output
quaternion_xx.dat

### Usage and parameters


--------------------------------------------------------------------------------
## make_data.c
### Input
None
### Output

### Usage and parameters


