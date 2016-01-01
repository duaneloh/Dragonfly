import numpy as np
import ConfigParser
import argparse
import sys
import os
import time
from collections import OrderedDict
from scipy.interpolate import interp1d

################################################################################
# Functions to read config file
################################################################################

def extract_param(config_file, section, tag):
    config      = ConfigParser.ConfigParser()
    config.read(config_file)
    return config.get(section, tag)

def read_detector_config(config_file, show=False):
    config      = ConfigParser.ConfigParser()
    config.read(config_file)
    params      = OrderedDict()
    params['wavelength']  = config.getfloat('parameters', 'lambda')
    params['detd']        = config.getfloat('parameters', 'detd')
    params['detsize']     = config.getint('parameters', 'detsize')
    params['pixsize']     = config.getfloat('parameters', 'pixsize')
    params['stoprad']     = config.getfloat('parameters', 'stoprad')
    if show:
        for k,v in params.items():
            print '{:<15}:{:10.4f}'.format(k, v)
    return params

def compute_q_params(det_dist, det_size, pix_size, in_wavelength, show=False, squareDetector=True):
    """
    Resolution computed in inverse Angstroms, crystallographer's convention
    In millimeters: det_dist, det_size, pix_size
    In Angstroms:   in_wavelength

    """
    det_max_half_len = pix_size * int((det_size-1)/2.)
    params      = OrderedDict()
    if squareDetector:
        max_angle   = np.arctan(np.sqrt(2.) * det_max_half_len / det_dist)
    else:
        max_angle   = np.arctan(det_max_half_len / det_dist)
    min_angle   = np.arctan(pix_size / det_dist)
    q_max       = 2. * np.sin(0.5 * max_angle) / in_wavelength
    q_sep       = 2. * np.sin(0.5 * min_angle) / in_wavelength
    fov_in_A    = 1. / q_sep
    half_p_res  = 0.5 / q_max
    params['max_angle'] = max_angle
    params['min_angle'] = min_angle
    params['q_max']     = q_max
    params['q_sep']     = q_sep
    params['fov_in_A']  = fov_in_A
    params['half_p_res']= half_p_res

    if show:
        for k,v in params.items():
            print '{:<15}:{:10.4f}'.format(k, v)
        print '{:<15}:{:10.4f}'.format("voxel-length of reciprocal volume", fov_in_A/half_p_res)
    return params

################################################################################
# Functions to read and process pdb
################################################################################

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
    return out_atoms[1:,:,:].reshape(-1,5)

def atoms_to_density_map(atoms, voxelSZ):
    (x, y, z) = atoms[:,1:4].T.copy()
    (x_min, x_max) = (x.min(), x.max())
    (y_min, y_max) = (y.min(), y.max())
    (z_min, z_max) = (z.min(), z.max())

    grid_len = max([x_max - x_min, y_max - y_min, z_max - z_min])
    R = np.int(np.ceil(grid_len / voxelSZ))
    if R % 2 == 0:
        R += 1

    elec_den = atoms[:,0].copy()

    x = (x-x_min)/voxelSZ
    y = (y-y_min)/voxelSZ
    z = (z-z_min)/voxelSZ

    bins = np.arange(R+1)
    all_bins = np.vstack((bins,bins,bins))
    coords = np.asarray([x,y,z]).T
    (h, h_edges) = np.histogramdd(coords, bins=all_bins, weights=elec_den)
    return h

def low_pass_filter_density_map(in_arr, damping=-2., thr=1.E-3, num_cycles=2):
    (xl,yl,zl) = in_arr.shape
    (xx,yy,zz) = np.mgrid[-1:1:xl*1j, -1:1:yl*1j, -1:1:zl*1j]
    fil = np.fft.ifftshift(np.exp(damping*(xx*xx + yy*yy + zz*zz)))
    out_arr = in_arr.copy()
    for i in range(num_cycles):
        ft = fil*np.fft.fftn(out_arr)
        out_arr = np.real(np.fft.ifftn(ft))
        out_arr *= (out_arr > thr)
    return out_arr

def write_density_to_file(in_den_file, in_den):
    with open(in_den_file, "w") as fp:
        for l0 in in_den:
            for l1 in l0:
                tmp = ' '.join(l1.astype('str'))
                fp.write(tmp + '\n')

################################################################################
# Script begins
################################################################################
class my_timer(object):
    def __init__(self):
        self.t0 = time.time()
        self.ts = self.t0

    def reset(self):
        t1 = time.time()
        self.t0 = t1

    def reset_and_report(self, msg):
        t1 = time.time()
        print "{:-<30}:{:5.5f} seconds".format(msg, t1-self.t0)
        self.t0 = t1

    def report_time_since_beginning(self):
        print "="*80
        print "{:-<30}:{:5.5f} seconds".format("Since beginning", time.time() - self.ts)

if __name__ == "__main__":

    timer       = my_timer()
    parser      = argparse.ArgumentParser(description="make electron density")
    parser.add_argument(dest='config_file')
    parser.add_argument("-v", "--verbose", dest="vb", action="store_true", default=False)
    parser.add_argument("-m", "--main_dir", dest="main_dir", help="relative path to main repository directory\n(where data aux utils are stored)")
    args        = parser.parse_args()
    args.main_dir = args.main_dir if args.main_dir else os.path.dirname(args.config_file)

    pm          = read_detector_config(args.config_file, show=args.vb)
    pdb_file    = os.path.join(args.main_dir, extract_param(args.config_file, 'files', "pdb"))
    q_pm        = compute_q_params(pm['detd'], pm['detsize'], pm['pixsize'], pm['wavelength'], show=args.vb)
    t1          = time.time()
    timer.reset_and_report("Reading detector") if args.vb else timer.reset()

    fov_len     = int(np.ceil(q_pm['fov_in_A']/q_pm['half_p_res']) + 1)
    eV          = wavelength_in_A_to_eV(pm['wavelength'])
    aux_dir     = os.path.join(args.main_dir, extract_param(args.config_file, 'files', "scatt_dir"))
    atom_types  = find_atom_types_in_pdb(pdb_file)
    scatt_list  = make_scatt_list(atom_types, aux_dir, eV)
    atoms       = read_atom_coords_from_pdb(pdb_file, scatt_list)
    (s_l, t_l)  = read_symmetry_from_pdb(pdb_file)
    all_atoms   = apply_symmetry(atoms, s_l, t_l)
    timer.reset_and_report("Reading PDB") if args.vb else timer.reset()

    den         = atoms_to_density_map(all_atoms, q_pm['half_p_res'])
    lp_den      = low_pass_filter_density_map(den)
    timer.reset_and_report("Creating density map") if args.vb else timer.reset()

    den_file    = os.path.join(args.main_dir, extract_param(args.config_file, 'files', "density_file"))
    write_density_to_file(den_file, lp_den)
    timer.reset_and_report("Writing densities to file") if args.vb else timer.reset()

    timer.report_time_since_beginning() if args.vb else timer.reset()
