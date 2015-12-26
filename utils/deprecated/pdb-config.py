import numpy as np
import sys
import subprocess
import os
import time
import ConfigParser

config = ConfigParser.ConfigParser()
config.read('config.ini')

pdbFile = config.get('files', 'pdb')
wavelength = config.getfloat('parameters', 'lambda')
detd = config.getfloat('parameters', 'detd')
detsize = config.getint('parameters', 'detsize')
pixsize = config.getfloat('parameters', 'pixsize')
qmax = 2. / wavelength * np.sin(0.5 * np.arctan(detsize/2./detd))
resolution = 1. / qmax

print pdbFile, wavelength, detd, detsize, qmax, resolution

# generate contrast from PDB file
cmd = 'python utils/get_pdb_coor.py ' + pdbFile + ' ' + str(resolution)
os.system(cmd)
#p = subprocess.Popen(cmd, shell=True)
#p.communicate()
#sys.exit()


# apply a low-pass filter to the contrast to avoid ring artifacts
cmd = 'python low_pass_filter.py'
os.system(cmd)
#p = subprocess.Popen(cmd, shell=True)
#p.communicate()
print 'finish generating contrast\n'


# generate intensity distribution
#cmd = 'gcc make_intensity.c -O3 -lm -lfftw3 -o intens'
#p = subprocess.Popen(cmd, shell=True)
#p.communicate()
#cmd = './make_intensity filtered-DensityMap.dat ' + str(qmax)
#p = subprocess.Popen(cmd, shell=True)
#p.communicate()
#p = subprocess.Popen('rm intens', shell=True)
#p.communicate()
#print 'finish generating intensity distribution\n'
#
#
#
## generate detector.dat
#fin = open('DensityMap.dat', 'r')
#R = np.int(fin.readline().split()[0])
#fin.close()
#sigma = np.double(qmax) / R        # oversampling ratio
#cmd = 'gcc make_detector.c -O3 -lm -o det'
#p = subprocess.Popen(cmd, shell=True)
#p.communicate()
#cmd = './det ' + str(L) + ' ' + str(qmax) + ' ' + str(sigma) + ' ' + str(D)
#p = subprocess.Popen(cmd, shell=True)
#p.communicate()
#p = subprocess.Popen('rm det', shell=True)
#p.communicate()
#print 'finish generating detector geometry\n'
