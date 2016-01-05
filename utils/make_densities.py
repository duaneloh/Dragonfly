import numpy as np
import ConfigParser
import argparse
import sys
import os
from collections import OrderedDict
from scipy.interpolate import interp1d
from py_src import read_config
from py_src import process_pdb 
from py_src import py_utils 

if __name__ == "__main__":

    timer       = py_utils.my_timer()
    parser      = argparse.ArgumentParser(description="make electron density")
    parser.add_argument(dest='config_file')
    parser.add_argument("-v", "--verbose", dest="vb", action="store_true", default=False)
    parser.add_argument("-m", "--main_dir", dest="main_dir", help="relative path to main repository directory\n(where data aux utils are stored)")
    args        = parser.parse_args()
    args.main_dir = args.main_dir if args.main_dir else os.path.dirname(args.config_file)

    pm          = read_config.get_detector_config(args.config_file, show=args.vb)
    pdb_file    = os.path.join(args.main_dir, read_config.get_param(args.config_file, 'make_densities', "pdb"))
    q_pm        = read_config.compute_q_params(pm['detd'], pm['detsize'], pm['pixsize'], pm['wavelength'], show=args.vb)
    t1          = time.time()
    timer.reset_and_report("Reading detector") if args.vb else timer.reset()

    fov_len     = int(np.ceil(q_pm['fov_in_A']/q_pm['half_p_res']) + 1)
    eV          = process_pdb.wavelength_in_A_to_eV(pm['wavelength'])
    aux_dir     = os.path.join(args.main_dir, read_config.get_param(args.config_file, 'make_densities', "scatt_dir"))
    atom_types  = process_pdb.find_atom_types(pdb_file)
    scatt_list  = process_pdb.make_scatt_list(atom_types, aux_dir, eV)
    atoms       = process_pdb.get_atom_coords(pdb_file, scatt_list)
    (s_l, t_l)  = process_pdb.read_symmetry(pdb_file)
    all_atoms   = process_pdb.apply_symmetry(atoms, s_l, t_l)
    timer.reset_and_report("Reading PDB") if args.vb else timer.reset()

    den         = process_pdb.atoms_to_density_map(all_atoms, q_pm['half_p_res'])
    lp_den      = process_pdb.low_pass_filter_density_map(den)
    timer.reset_and_report("Creating density map") if args.vb else timer.reset()

    den_file    = os.path.join(args.main_dir, read_config.get_param(args.config_file, 'make_densities', "density_file"))
    process_pdb.write_density_to_file(den_file, lp_den)
    timer.reset_and_report("Writing densities to file") if args.vb else timer.reset()

    timer.report_time_since_beginning() if args.vb else timer.reset()
