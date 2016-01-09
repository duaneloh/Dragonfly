## Quick start  
### Spawn a reconstruction directory
Type in the topmost directory:
	```
	./init_new_recon.py
	```.
This makes the binaries needed for the package and links them into a new reconstruction directory named "recon_xxxx". For help on customizing this setup script (e.g. custom folder name or counter):
	```
	./init_new_recon.py -h
	```.

### Configure experiment
Go to your newly created recon directory
	```
	cd recon_xxxx
	```.
Change the experiment parameters in config.ini to your liking. 
Here are some things that you might like to change:
- [in_pdb_file] relative path to your own PDB file
- scattering setup (detector distance, photon wavelength, etc)
- [num_data] the number of diffraction patterns
- [mean_count] the average number of photons per pattern
- [log_file] name of log file

When ready to start creating synthetic data, type:
	```
	./sim_setup.py
	```.
Again, you can get help to customize this using the command:
	```
	./sim_setup.py -h
	```.

### Start your EMC reconstruction
You can start a single process reconstruction in the recon directory this way:
	```
	./emc <num_iterations> <path to config file> [threads per process]
	```,
where ```<necessary arguments>``` and ```[optional arguments]```. The default number of threads per process is defined by the system parameter OMP_NUM_THREADS.
By default, the intermediate output of the reconstruction is stored in the recon directory's data subdirectory.

To spawn multiple MPI reconstructions from your recon directory:
	```
	mpirun -n <num_mpi_processes> ./emc <num_iterations> <path to config file> [threads per process]
	```.
By default, images from the reconstruction that are generated by this script are saved to the data/images subdirectory.

### Monitor the progress of your reconstruction 
In the recon directory type:
	```
	./autoplot.py
	```.
Checking the "keep checking" box will automagically look for new reconstructed 3D volumes in the default data directories.


