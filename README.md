# EMC
--------------------------------------------------------------------------------
## exp_config.dat
Structure of exp_config.dat

--------------------------------------------------------------------------------
## make_densities.py
Create electron density map from PDB file
### Input
exp_config.dat
### Output
densities.dat

Densities.dat is a flattened ASCII record of the floating point numbers of a 3D cubic electron density distribution. 

### Usage and parameters
This is a Python 2.7 module that requires the following non-standard packages:
numpy
scipy
To get usage help:
	python make_densities.py -h
To start making densities from experimental parameters specified in the config file "exp_config.dat"
	python make_densities.py <exp_config.dat>

--------------------------------------------------------------------------------
## make_intensities.py 
Create 3D intensity map from electron density
### Input
densities.dat
### Output
intensities.bin

### Usage and parameters


--------------------------------------------------------------------------------
## make_detector.c
This module generates the detector.dat file which contains information about
each detector pixel. 
### Input
exp_config.dat
### Output
detector.dat

The detector.dat file is an ASCII (human-readable) file.
 - The first line contains a single integer which is the number of pixels
 - From the second line on, there are 6 columns per line. Each line represents
   information about a pixel. The pixel index in the rest of this package
   refers to the line number in this file.
 - The first three columns give the 3D reciprocal space coordinates of the pixel
   relative to the 3D intensity model.
 - The fourth column is a multiplicative factor for polarization and solid angle
   corrections for the pixel.
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


