import numpy as np
import sys

job = sys.argv[1]
voxelSZ = np.double(sys.argv[2])	# angstrom

fin = open(job,"r")

mf0_C = [6., 12.]
mf0_N = [7., 14.]
mf0_O = [8., 16.]
mf0_S = [16., 32.]
mf0_P = [15., 31.]
mf0_Cu = [29., 63.5]

def append_atom(atomlist, atom, line):
	atomlist.append([atom[0], 
	                float(line[30:38].strip()), 
					float(line[38:46].strip()), 
					float(line[46:54].strip()),
					atom[1]])

tmp_atoms = []
atom_count = 0
line = fin.readline().strip()
while (line):

	if line[0:4] == "ATOM" or line[0:6] == "HETATM":
		atom_count += 1
		# occupany > 50 % || one of either if occupany = 50 %
		if ( (np.double(line[56:60]) > 0.5) or (np.double(line[56:60]) == 0.5 and line[16] != "B") ):
			if (line[-1] == "C"):
				append_atom(tmp_atoms, mf0_C, line)
			elif (line[-1] == "O"):
				append_atom(tmp_atoms, mf0_O, line)
			elif (line[-1] == "N"):
				append_atom(tmp_atoms, mf0_N, line)
			elif (line[-1] == "S"):
				append_atom(tmp_atoms, mf0_S, line)
			elif (line[-1] == "P"):
				append_atom(tmp_atoms, mf0_P, line)
			elif (line[-2:] == "CU"):
				append_atom(tmp_atoms, mf0_Cu, line)
			else:
				s = line[-2:] + " not in the current atom list"
				print s
	
	line = fin.readline().strip() 
	sys.stderr.write('\rFound %d atoms' % atom_count)

fin.close()
sys.stderr.write('\n')

atoms = np.array(tmp_atoms)
print atoms.shape, atoms[0]

# apply symmetry operations:
fin = open(job,"r")
line = fin.readline().strip()
sym_list = []
trans_list = []
while (line):
	if (line[13:18] == "BIOMT"):
		sym_list.append( [np.double(line[24:33]), np.double(line[34:43]), np.double(line[44:53])] )
		trans_list.append(np.double(line[58:68]))
	line = fin.readline().strip()
fin.close()

for i in range(len(sym_list)/3):
	sym_op = np.zeros((3,3))	
	trans = np.zeros(3)	
	for j in range(3):
		for k in range(3):
			sym_op[j][k] = sym_list[3*i+j][k]
		trans[j] = trans_list[3*i+j]
	vecs = sym_op.dot(atoms[:atom_count,1:4].T).T + trans
	f0s = atoms[:atom_count,0].reshape(atom_count,1)
	ms = atoms[:atom_count,4].reshape(atom_count,1)
	vecs = np.concatenate((f0s, vecs, ms), axis=1)
	atoms = np.append(atoms, vecs, axis=0)

x_max = atoms[:,1].max()
x_min = atoms[:,1].min()
y_max = atoms[:,2].max()
y_min = atoms[:,2].min()
z_max = atoms[:,3].max()
z_min = atoms[:,3].min()

total_charge = atoms[:,0].sum()
weight = atoms[:,4].sum()

r = max([x_max - x_min, y_max - y_min, z_max - z_min]) / 2
R = np.int(np.ceil(r / voxelSZ))

fout = open("emc.log", "a")
tmp = "PDB file: " + job + ":\n"
tmp += "total charge = %.1f e\n" % total_charge
tmp += "weight = %.1f kDa\n" % (weight/1.e3)
tmp += "R = %d\n" % R
tmp += "radius = %.3f nm\n" % (r/10)
tmp += "resolution = %.3f nm\n\n" % (voxelSZ/10)
fout.write(tmp)
fout.close()
sys.exit()

# center atom coordinates
DenMap = np.zeros((2*R+1, 2*R+1, 2*R+1))
atoms[:,1] -= (x_max + x_min)/2
atoms[:,2] -= (y_max + y_min)/2
atoms[:,3] -= (z_max + z_min)/2

idx = 0
idy = 0
idz = 0

for i in range(len(atoms)):
	if atoms[i][1] > 0:
		idx = int(np.ceil((atoms[i][1] - voxelSZ/2)/voxelSZ)) + R
	if atoms[i][2] > 0:
		idy = int(np.ceil((atoms[i][2] - voxelSZ/2)/voxelSZ)) + R
	if atoms[i][3] > 0:
		idz = int(np.ceil((atoms[i][3] - voxelSZ/2)/voxelSZ)) + R
	if atoms[i][1] < 0:
		idx = int(np.floor((atoms[i][1] + voxelSZ/2)/voxelSZ)) + R
	if atoms[i][2] < 0:
		idy = int(np.floor((atoms[i][2] + voxelSZ/2)/voxelSZ)) + R
	if atoms[i][3] < 0:
		idz = int(np.floor((atoms[i][3] + voxelSZ/2)/voxelSZ)) + R
	DenMap[idx][idy][idz] += atoms[i][0]

fout = open("DensityMap.dat","w")
tmp = str(R) + "\n"
fout.write(tmp)
for i in range(2*R+1):
	for j in range(2*R+1):
		for k in range(2*R+1):
			tmp = str(DenMap[i][j][k]) + "\n"
			fout.write(tmp)
fout.close()
