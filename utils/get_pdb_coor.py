import numpy as np
import sys

job = sys.argv[1]
voxelSZ = np.double(sys.argv[2])	# angstrom

fin = open(job,"r")

f0_C = 6.0
f0_N = 7.0
f0_O = 8.0
f0_S = 16.0
f0_P = 15.0
f0_Cu = 29.0

m_C = 12.0
m_N = 14.0
m_O = 16.0
m_S = 32.0
m_P = 31.0
m_Cu = 63.5

tmp_atoms = []
atom_count = 0
line = fin.readline().strip()
while (line):

	if (line[0:4] == "ATOM"):
		atom_count += 1
		# occupany > 50 % || one of either if occupany = 50 %
		if ( (np.double(line[56:60]) > 0.5) or (np.double(line[56:60]) == 0.5 and line[16] != "B") ):
			if (line[-1] == "C"):
				tmp_atoms.append([f0_C, line[30:38].strip(), line[38:46].strip(), line[46:54].strip(), m_C])
			elif (line[-1] == "O"):
				tmp_atoms.append([f0_O, line[30:38].strip(), line[38:46].strip(), line[46:54].strip(), m_O])
			elif (line[-1] == "N"):
				tmp_atoms.append([f0_N, line[30:38].strip(), line[38:46].strip(), line[46:54].strip(), m_N])
			elif (line[-1] == "S"):
				tmp_atoms.append([f0_S, line[30:38].strip(), line[38:46].strip(), line[46:54].strip(), m_S])
			elif (line[-1] == "P"):
				tmp_atoms.append([f0_P, line[30:38].strip(), line[38:46].strip(), line[46:54].strip(), m_P])
			elif (line[-2:] == "CU"):
				tmp_atoms.append([f0_Cu, line[30:38].strip(), line[38:46].strip(), line[46:54].strip(), m_Cu])
			else:
				s = line[-2:] + " not in the current atom list"
				print s

	if (line[0:6] == "HETATM"):
		atom_count += 1
		# occupany > 50 % || one of either if occupany = 50 %
		if ( (np.double(line[56:60]) > 0.5) or (np.double(line[56:60]) == 0.5 and line[16] != "B") ):
			if (line[-1] == "C"):
				tmp_atoms.append([f0_C, line[30:38].strip(), line[38:46].strip(), line[46:54].strip(), m_C])
			elif (line[-1] == "O"):
				tmp_atoms.append([f0_O, line[30:38].strip(), line[38:46].strip(), line[46:54].strip(), m_O])
			elif (line[-1] == "N"):
				tmp_atoms.append([f0_N, line[30:38].strip(), line[38:46].strip(), line[46:54].strip(), m_N])
			elif (line[-1] == "S"):
				tmp_atoms.append([f0_S, line[30:38].strip(), line[38:46].strip(), line[46:54].strip(), m_S])
			elif (line[-1] == "P"):
				tmp_atoms.append([f0_P, line[30:38].strip(), line[38:46].strip(), line[46:54].strip(), m_P])
			elif (line[-2:] == "CU"):
				tmp_atoms.append([f0_Cu, line[30:38].strip(), line[38:46].strip(), line[46:54].strip(), m_Cu])
			else:
				s = line[-2:] + " not in the current atom list"
				print s

	line = fin.readline().strip() 

fin.close()

atoms = []
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
	for idx in range(len(tmp_atoms)):
		r_vec = np.zeros(3)
		vec = [float(tmp_atoms[idx][1]), float(tmp_atoms[idx][2]), float(tmp_atoms[idx][3])]
		for j in range(3):
			for k in range(3):
				r_vec[j] += sym_op[j][k]*vec[k]
		for j in range(3):
			r_vec[j] += trans[j]
		atoms.append([tmp_atoms[idx][0], r_vec[0], r_vec[1], r_vec[2], tmp_atoms[idx][4]]) 

x_max = float(atoms[0][1])
x_min = float(atoms[0][1])
y_max = float(atoms[0][2])
y_min = float(atoms[0][2])
z_max = float(atoms[0][3])
z_min = float(atoms[0][3])

total_charge = 0
weight = 0
for i in range(len(atoms)):
	if x_max < float(atoms[i][1]):
		x_max = float(atoms[i][1])
	if x_min > float(atoms[i][1]):
		x_min = float(atoms[i][1])
	if y_max < float(atoms[i][2]):
		y_max = float(atoms[i][2])
	if y_min > float(atoms[i][2]):
		y_min = float(atoms[i][2])
	if z_max < float(atoms[i][3]):
		z_max = float(atoms[i][3])
	if z_min > float(atoms[i][3]):
		z_min = float(atoms[i][3])
	total_charge += atoms[i][0]
	weight += atoms[i][4]

r = max(x_max - x_min, max(y_max - y_min, z_max - z_min)) / 2
R = np.int(np.ceil(r / voxelSZ))

fout = open("emc.log", "a")
tmp = "PDB file: " + job[0:4] + ":\n"
tmp += "total charge = %.1f e\n" % total_charge
tmp += "weight = %.1f amu\n" % weight
tmp += "R = %d\n" % R
tmp += "radius = %.3f nm\n" % (r/10)
tmp += "resolution = %.3f nm\n\n" % (voxelSZ/10)
fout.write(tmp)
fout.close()

# symmetrize atom coordinates
DenMap = np.zeros((2*R+1, 2*R+1, 2*R+1))
for i in range(len(atoms)):
	atoms[i][1] = float(atoms[i][1]) - (x_max + x_min)/2
	atoms[i][2] = float(atoms[i][2]) - (y_max + y_min)/2
	atoms[i][3] = float(atoms[i][3]) - (z_max + z_min)/2

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
