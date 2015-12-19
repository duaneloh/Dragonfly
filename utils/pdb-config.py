import numpy as np
import subprocess
import time

# user defined parameters
pdbFile          = '4BED.pdb'
PhotonEnergy     = 2.0  	# keV
PixelSize        = 100.0	# micron
DetectorDistance = 50.0 	# mm
L                = 50   	# radius of the largest disk that fits inside the actual detector (in pixels)



# calculate corresponding parameters for generating detector.dat
D = DetectorDistance/(PixelSize*1.e-3)					# detector distance in pixels
theta = np.arctan(L/D)							# maximum scattering angle
wavelength = 12.4/PhotonEnergy						# angstrom
resolution = wavelength/(4*np.sin(theta/2))				# angstrom
qmax = np.int(np.ceil(L*np.cos(theta)/np.cos(theta/2)))			# 2*qmax + 1: number of frequency sampling

fout = open('emc.log', 'w')
tmp = "D = %.3f pixels\ntheta = %.3f\nwavelength = %.3f angstrom\nqmax = %d\n\n" % (D, theta, wavelength, qmax)
fout.write(tmp)
fout.close()



# generate contrast from PDB file
cmd = 'python get_pdb_coor.py ' + pdbFile + ' ' + str(resolution)
p = subprocess.Popen(cmd, shell=True)
p.communicate()



# apply a low-pass filter to the contrast to avoid ring artifacts
cmd = 'python low_pass_filter.py'
p = subprocess.Popen(cmd, shell=True)
p.communicate()
print 'finish generating contrast\n'



# generate intensity distribution
cmd = 'gcc make_intensity.c -O3 -lm -lfftw3 -o intens'
p = subprocess.Popen(cmd, shell=True)
p.communicate()
cmd = './intens filtered-DensityMap.dat ' + str(qmax)
p = subprocess.Popen(cmd, shell=True)
p.communicate()
p = subprocess.Popen('rm intens', shell=True)
p.communicate()
print 'finish generating intensity distribution\n'



# generate detector.dat
fin = open('DensityMap.dat', 'r')
R = np.int(fin.readline().split()[0])
fin.close()
sigma = np.double(qmax) / R		# oversampling ratio
cmd = 'gcc make_detector.c -O3 -lm -o det'
p = subprocess.Popen(cmd, shell=True)
p.communicate()
cmd = './det ' + str(L) + ' ' + str(qmax) + ' ' + str(sigma) + ' ' + str(D)
p = subprocess.Popen(cmd, shell=True)
p.communicate()
p = subprocess.Popen('rm det', shell=True)
p.communicate()
print 'finish generating detector geometry\n'
