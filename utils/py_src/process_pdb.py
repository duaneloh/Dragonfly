'''Various functions to read PDB files for Dragonfly'''

from __future__ import print_function
from builtins import range
import sys
import logging
import os
from collections import OrderedDict
from six.moves.urllib.request import urlopen
import numpy as np
from scipy.interpolate import interp1d
try:
    import pyfftw
    WITH_PYFFTW = True
except ImportError:
    sys.stderr.write('No PyFFTW. FFTs will be slow\n')
    WITH_PYFFTW = False

def fetch_pdb(pdb_code):
    '''Get PDB file from aux directory if available, else download from RCSB'''
    print(pdb_code)
    if os.path.isfile('aux/%s.pdb' % (pdb_code.upper())):
        pass
    else:
        pdb_string = urlopen('http://www.rcsb.org/pdb/files/%s.pdb' % pdb_code.upper())
        with open('aux/%s.pdb' % pdb_code.upper()) as fptr:
            fptr.write(pdb_string)

def _find_atom_types(pdb_file):
    atoms = []
    with open(pdb_file) as fin:
        for line in fin:
            line = line.strip()
            if line[0:4] == "ATOM" or line[0:6] == "HETATM":
                atom_label = line[76:78].lstrip()
                if atom_label not in atoms:
                    atoms.append(atom_label)
    return atoms

def _interp_scattering(aux_dir, elem):
    with open(os.path.join(aux_dir, elem.lower()+".nff")) as fptr:
        lines = [l.strip().split() for l in fptr.readlines()]
        arr = np.asarray(lines[1:]).astype('float')
        energy, scatt_f0, scatt_f1 = arr.T
        i_f0 = interp1d(energy, scatt_f0, kind='linear')
        i_f1 = interp1d(energy, scatt_f1, kind='linear')
    return i_f0, i_f1

def _find_mass(aux_dir, elem):
    with open(os.path.join(aux_dir, "atom_mass.txt")) as fptr:
        lines = [l.strip().split() for l in fptr.readlines()]
        for line, mass in lines:
            if line.lower() == elem.lower():
                return float(mass)
    return None

def _make_scatt_list(atom_types, aux_dir, energy):
    scatt_list = OrderedDict()
    for elem in atom_types:
        scatt_f0, _ = _interp_scattering(aux_dir, elem)
        mass = _find_mass(aux_dir, elem)
        scatt_list[elem.upper()] = [float(scatt_f0(energy)), mass]
    return scatt_list

def _wavelength_in_A_to_eV(wavelength_in_A): # pylint: disable=C0103
    return 12398.419 / wavelength_in_A

def _append_atom(atomlist, atom, pdb_line):
    atomlist.append([atom[0],
                     float(pdb_line[30:38].strip()),
                     float(pdb_line[38:46].strip()),
                     float(pdb_line[46:54].strip()),
                     atom[1]])

def _get_atom_coords(pdb_file, scatt):
    tmp_atoms = []
    with open(pdb_file) as fin:
        for line in fin:
            line = line.strip()
            if line[0:4] == "ATOM" or line[0:6] == "HETATM":
                # occupany > 50 % || one of either if occupany = 50 %
                (occ, tag) = (float(line[56:60]), line[16])
                if (occ > 0.5) | ((occ == 0.5) & (tag != "B")):
                    atom_label = line[76:78].lstrip().upper()
                    if atom_label in scatt:
                        _append_atom(tmp_atoms, scatt[atom_label], line)
                    else:
                        logstr = line[76:78] + " not in the current atom list"
                        logging.info(logstr)
    return np.asarray(tmp_atoms)

def _read_symmetry(pdb_file):
    '''First symmetry operation is identity, followed by non-trivial symmetries'''
    sym_list = []
    trans_list = []
    with open(pdb_file) as fin:
        for line in fin:
            line = line.strip()
            if line[13:18] == "BIOMT":
                sym_list.append([float(line[24:33]), float(line[34:43]), float(line[44:53])])
                trans_list.append(float(line[58:68]))
    sym_arr = np.asarray(sym_list).reshape(-1, 3, 3)
    trans_arr = np.asarray(trans_list).reshape(-1, 3)
    return sym_arr, trans_arr

def _apply_symmetry(atoms, sym_list, trans_list):
    if len(sym_list) == 0:
        return atoms
    org_atoms = atoms[:, 1:4].T.copy()
    f0s = np.asarray([atoms[:, 0]]).T.copy()
    mass = np.asarray([atoms[:, 4]]).T.copy()
    total_ms = len(sym_list)*np.sum(mass) / 1.0e6
    logging.info("Mass of particle (MDa), %.3f", total_ms)
    out_atoms = np.zeros((len(sym_list),)+atoms.shape)
    for i, sym_op in enumerate(sym_list):
        trans = trans_list[i]
        vecs = sym_op.dot(org_atoms).T + trans
        to_app = np.concatenate((f0s, vecs, mass), axis=1)
        out_atoms[i] = to_app.copy()
    return out_atoms.reshape(-1, 5)

def atoms_to_density_map(atoms, voxel_size):
    '''Create electron density map from atom coordinate list'''
    (x, y, z) = atoms[:, 1:4].T.copy()
    (x_min, x_max) = (x.min(), x.max())
    (y_min, y_max) = (y.min(), y.max())
    (z_min, z_max) = (z.min(), z.max())

    grid_len = max([x_max - x_min, y_max - y_min, z_max - z_min])
    r_val = np.int(np.ceil(grid_len / voxel_size))
    if r_val % 2 == 0:
        r_val += 1
    logging.info("Length of particle (voxels), %d", r_val)
    elec_den = atoms[:, 0].copy()

    x = (x-0.5*(x_max+x_min-grid_len))/voxel_size
    y = (y-0.5*(y_max+y_min-grid_len))/voxel_size
    z = (z-0.5*(z_max+z_min-grid_len))/voxel_size

    bins = np.arange(r_val+1)
    all_bins = np.vstack((bins, bins, bins))
    coords = np.asarray([x, y, z])
    integ = np.floor(coords)
    frac = coords - integ
    ix, iy, iz = tuple(integ) # pylint: disable=C0103
    fx, fy, fz = tuple(frac) # pylint: disable=C0103
    cx, cy, cz = 1.-fx, 1.-fy, 1.-fz # pylint: disable=C0103
    h_total = np.histogramdd(np.asarray([ix, iy, iz]).T,
                             weights=elec_den*cx*cy*cz, bins=all_bins)[0]
    h_total += np.histogramdd(np.asarray([ix, iy, iz+1]).T,
                              weights=elec_den*cx*cy*fz, bins=all_bins)[0]
    h_total += np.histogramdd(np.asarray([ix, iy+1, iz]).T,
                              weights=elec_den*cx*fy*cz, bins=all_bins)[0]
    h_total += np.histogramdd(np.asarray([ix, iy+1, iz+1]).T,
                              weights=elec_den*cx*fy*fz, bins=all_bins)[0]
    h_total += np.histogramdd(np.asarray([ix+1, iy, iz]).T,
                              weights=elec_den*fx*cy*cz, bins=all_bins)[0]
    h_total += np.histogramdd(np.asarray([ix+1, iy, iz+1]).T,
                              weights=elec_den*fx*cy*fz, bins=all_bins)[0]
    h_total += np.histogramdd(np.asarray([ix+1, iy+1, iz]).T,
                              weights=elec_den*fx*fy*cz, bins=all_bins)[0]
    h_total += np.histogramdd(np.asarray([ix+1, iy+1, iz+1]).T,
                              weights=elec_den*fx*fy*fz, bins=all_bins)[0]
    return h_total

def low_pass_filter_density_map(in_arr, damping=-1., thr=1.E-3, num_cycles=2, threads=4):
    '''Convolve density map by Gaussian with given damping coefficient'''
    xl, yl, zl = in_arr.shape # pylint: disable=C0103
    xx, yy, zz = np.mgrid[-1:1:xl*1j, -1:1:yl*1j, -1:1:zl*1j] # pylint: disable=C0103
    fil = np.fft.ifftshift(np.exp(damping*(xx*xx + yy*yy + zz*zz)))
    out_arr = in_arr.copy()
    if WITH_PYFFTW:
        for _ in range(num_cycles):
            ft_arr = fil * pyfftw.interfaces.numpy_fft.fftn(out_arr,
                                                            planner_effort='FFTW_ESTIMATE',
                                                            threads=threads)
            out_arr = np.real(pyfftw.interfaces.numpy_fft.ifftn(ft_arr,
                                                                planner_effort='FFTW_ESTIMATE',
                                                                threads=threads))
            out_arr *= (out_arr > thr)
    else:
        for _ in range(num_cycles):
            ft_arr = fil * np.fft.fftn(out_arr)
            out_arr = np.real(np.fft.ifftn(ft_arr))
            out_arr *= (out_arr > thr)
    return out_arr.copy()

def process(pdb_file, aux_dir, wavelength):
    '''Get atom scattering list from PDB file
    Generates list of coordinates, scattering f1 and mass for each atom
    '''
    energy = _wavelength_in_A_to_eV(wavelength)
    atom_types = _find_atom_types(pdb_file)
    scatt_list = _make_scatt_list(atom_types, aux_dir, energy)
    atoms = _get_atom_coords(pdb_file, scatt_list)
    sym_l, trans_l = _read_symmetry(pdb_file)
    return _apply_symmetry(atoms, sym_l, trans_l)
