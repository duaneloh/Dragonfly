import numpy as np
import sys
import logging
import os
import time
from collections import OrderedDict
from scipy.interpolate import interp1d

def find_atom_types(pdb_file):
    atoms = []
    with open(pdb_file) as fin:
        for line in fin:
            line = line.strip()
            if line[0:4] == "ATOM" or line[0:6] == "HETATM":
                atom_label = line[-2:].lstrip()
                if atom_label not in atoms:
                    atoms.append(atom_label)
    return atoms

def interp_scattering(aux_dir, elem):
    with open(os.path.join(aux_dir, elem.lower()+".nff")) as fp:
        lines = [l.strip().split() for l in fp.readlines()]
        arr = np.asarray(lines[1:]).astype('float')
        (energy, f0, f1) = arr.T
        i_f0 = interp1d(energy, f0, kind='linear')
        i_f1 = interp1d(energy, f1, kind='linear')
    return (i_f0, i_f1,)

def find_mass(aux_dir, elem):
    with open(os.path.join(aux_dir, "atom_mass.txt")) as fp:
        lines = [l.strip().split() for l in fp.readlines()]
        for l, m in lines:
            if l.lower() == elem.lower():
                return float(m)

def make_scatt_list(atom_types, aux_dir, eV):
    scatt_list = OrderedDict()
    for elem in atom_types:
        (f0,f1,) = interp_scattering(aux_dir, elem)
        mass    = find_mass(aux_dir, elem)
        scatt_list[elem.upper()] = [float(f0(eV)), mass]
    return scatt_list

def wavelength_in_A_to_eV(wavelength_in_A):
    return 12398.419 / wavelength_in_A

def append_atom(atomlist, atom, pdb_line):
    atomlist.append([atom[0],
                    float(pdb_line[30:38].strip()),
                    float(pdb_line[38:46].strip()),
                    float(pdb_line[46:54].strip()),
                    atom[1]])

def get_atom_coords(pdb_file, scatt):
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
                        logging.info(s)
    return np.asarray(tmp_atoms)

def read_symmetry(pdb_file):
    """
    First symmetry operation is identity, followed by non-trivial symmetries

    """
    sym_list = []
    trans_list = []
    with open(pdb_file) as fin:
        for line in fin:
            line = line.strip()
            if (line[13:18] == "BIOMT"):
                sym_list.append([float(line[24:33]), float(line[34:43]), float(line[44:53])])
                trans_list.append(float(line[58:68]))
    sym_arr     = np.asarray(sym_list).reshape(-1,3,3)
    trans_arr   = np.asarray(trans_list).reshape(-1,3)
    return (sym_arr, trans_arr)

def apply_symmetry(atoms, sym_list, trans_list):
    num_atoms = len(atoms)
    org_atoms = atoms[:,1:4].T.copy()
    f0s = np.asarray([atoms[:,0]]).T.copy()
    ms = np.asarray([atoms[:,4]]).T.copy()
    total_ms = len(sym_list)*np.sum(ms) / 1.0e6
    msg = "Mass of particle (MDa), %.3f"%(total_ms)
    logging.info(msg)
    out_atoms = np.zeros((len(sym_list),)+atoms.shape)
    for i in xrange(len(sym_list)):
        sym_op = sym_list[i]
        trans = trans_list[i]
        vecs = sym_op.dot(org_atoms).T + trans
        to_app = np.concatenate((f0s, vecs, ms), axis=1)
        out_atoms[i] = to_app.copy()
    return out_atoms.reshape(-1,5)

def atoms_to_density_map(atoms, voxelSZ):
    (x, y, z) = atoms[:,1:4].T.copy()
    (x_min, x_max) = (x.min(), x.max())
    (y_min, y_max) = (y.min(), y.max())
    (z_min, z_max) = (z.min(), z.max())

    grid_len = max([x_max - x_min, y_max - y_min, z_max - z_min])
    R = np.int(np.ceil(grid_len / voxelSZ))
    if R % 2 == 0:
        R += 1
    msg = "Length of particle (voxels), %d"%(R)
    logging.info(msg)
    elec_den = atoms[:,0].copy()

    x = (x-x_min)/voxelSZ
    y = (y-y_min)/voxelSZ
    z = (z-z_min)/voxelSZ

    bins = np.arange(R+1)
    all_bins = np.vstack((bins,bins,bins))
    coords = np.asarray([x,y,z]).T
    (h, h_edges) = np.histogramdd(coords, bins=all_bins, weights=elec_den)
    return h

def low_pass_filter_density_map(in_arr, damping=-1., thr=1.E-3, num_cycles=2):
    (xl,yl,zl) = in_arr.shape
    (xx,yy,zz) = np.mgrid[-1:1:xl*1j, -1:1:yl*1j, -1:1:zl*1j]
    fil = np.fft.ifftshift(np.exp(damping*(xx*xx + yy*yy + zz*zz)))
    out_arr = in_arr.copy()
    for i in range(num_cycles):
        ft = fil*np.fft.fftn(out_arr)
        out_arr = np.real(np.fft.ifftn(ft))
        out_arr *= (out_arr > thr)
    return out_arr.copy()
