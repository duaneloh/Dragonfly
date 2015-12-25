import numpy as np
import ConfigParser
import check_exp_config

################################################################################
# Useful functions
################################################################################

scatt_list ={
    "C":[6., 12.],
    "N":[7., 14.],
    "O":[8., 16.],
    "S":[16., 32.],
    "P":[15., 31.],
    "CU":[29., 63.5]}

def find_atom_types_in_pdb(pdb_file):
    atoms = []
    with open(pdb_file) as fin:
        for line in fin:
            line = line.strip()
            if line[0:4] == "ATOM" or line[0:6] == "HETATM":
                atom_label = line[-2:].lstrip()
                if atom_label not in atoms:
                    atoms.append(atom_label)
    return atoms

def append_atom(atomlist, atom, pdb_line):
    atomlist.append([atom[0],
                    float(pdb_line[30:38].strip()),
                    float(pdb_line[38:46].strip()),
                    float(pdb_line[46:54].strip()),
                    atom[1]])

def read_atom_coords_from_pdb(pdb_file, scatt):
    tmp_atoms = []
    with open(pdb_file) as fin:
        for line in fin:
            line = line.strip()
            if line[0:4] == "ATOM" or line[0:6] == "HETATM":
                # occupany > 50 % || one of either if occupany = 50 %
                (occ, tag) = (float(line[56:60]), line[16])
                if ((occ > 0.5) | ((occ == 0.5) & (tag != "B"))):
                    atom_label = line[-2:].lstrip()
                    if atom_label in scatt:
                        append_atom(tmp_atoms, scatt[atom_label], line)
                    else:
                        s = line[-2:] + " not in the current atom list"
                        print s
    return np.asarray(tmp_atoms)

def read_symmetry_from_pdb(pdb_file):
    sym_list = []
    trans_list = []
    num_sym = 0
    with open(pdb_file) as fin:
        for line in fin:
            line = line.strip()
            if (line[13:18] == "BIOMT"):
                num_sym += 1
                sym_list.append([float(line[24:33]), float(line[34:43]), float(line[44:53])])
                trans_list.append(float(line[58:68]))
    print num_sym, "symmetries found"
    sym_arr     = np.asarray(sym_list).reshape(-1,3,3)
    trans_arr   = np.asarray(trans_list).reshape(-1,3)
    return (sym_arr, trans_arr)

def apply_symmetry(atoms, sym_list, trans_list):
    num_atoms = len(atoms)
    org_atoms = atoms[:,1:4].T.copy()
    f0s = np.asarray([atoms[:,0]]).T.copy()
    ms = np.asarray([atoms[:,4]]).T.copy()
    out_atoms = np.zeros((len(sym_list)+1,)+atoms.shape)
    out_atoms[0] = atoms.copy()
    for i in xrange(len(sym_list)):
        sym_op = sym_list[i]
        trans = trans_list[i]
        vecs = sym_op.dot(org_atoms).T + trans
        to_app = np.concatenate((f0s, vecs, ms), axis=1)
        out_atoms[i+1] = to_app.copy()
    return out_atoms.reshape(-1,5)

def atoms_to_density_map(atoms, voxelSZ, fov_len):
    (x, y, z) = atoms[:,1:4].T.copy()
    (x_min, x_max) = (x.min(), x.max())
    (y_min, y_max) = (y.min(), y.max())
    (z_min, z_max) = (z.min(), z.max())

    elec_den = atoms[:,0].copy()

    x = (x-x_min)/voxelSZ + 1
    y = (y-y_min)/voxelSZ + 1
    z = (z-z_min)/voxelSZ + 1

    bins = np.arange(fov_len)
    all_bins = np.vstack((bins,bins,bins))
    coords = np.asarray([x,y,z]).T
    (h, h_edges) = np.histogramdd(coords, bins=all_bins, weights=elec_den)
    return h

def low_pass_filter_density_map(in_arr, damping=-2.):
    (xl,yl,zl) = in_arr.shape
    #TODO: Need to check odd and even array sizes
    (xx,yy,zz) = np.mgrid[-1:1:xl*1j, -1:1:yl*1j, -1:1:zl*1j]
    fil = np.fft.ifftshift(np.exp(damping*(xx*xx + yy*yy + zz*zz)))
    ft = fil*np.fft.fftn(in_arr)
    return np.real(np.fft.ifftn(ft))

################################################################################
# Script begins
################################################################################

if __name__ == "__main__":

    pass
