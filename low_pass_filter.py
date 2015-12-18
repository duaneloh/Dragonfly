import numpy as np

job = 'DensityMap.dat'
fin = open(job, 'r')

R  = np.int(fin.readline().split()[0])
R2 = R**2

N = 2*R + 1
contrast = np.zeros((N, N, N))
for i in xrange(N):
	for j in xrange(N):
		for k in xrange(N):
			contrast[i][j][k] = np.double(fin.readline().split()[0])
fin.close()

damping = 1.0
F = np.fft.fftshift(np.fft.fftn(contrast))
for i in xrange(N):
	for j in xrange(N):
		for k in xrange(N):
			r2 = (i-R)**2 + (j-R)**2 + (k-R)**2
			F[i][j][k] *= np.exp(-damping*r2/R2)

contrast2 = np.fft.ifftn(np.fft.ifftshift(F))

job = 'filtered-DensityMap.dat'
fout = open(job, 'w')
tmp = str(N) + ' ' + str(N) + ' ' + str(N) + '\n'
fout.write(tmp)

for i in xrange(N):
	for j in xrange(N):
		for k in xrange(N):
			tmp = str(contrast2[i][j][k].real) + '\n'
			fout.write(tmp)
fout.close()
