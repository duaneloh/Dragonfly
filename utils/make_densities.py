#!/usr/bin/env python
import numpy as np
import argparse
import sys
import os
import logging
from py_src import read_config
from py_src import process_pdb
from py_src import py_utils

if __name__ == "__main__":
    # logging config must occur before my_argparser, because latter already starts logging
    logging.basicConfig(filename="recon.log", level=logging.INFO, format='%(asctime)s - %(levelname)s -%(message)s')
    parser      = py_utils.my_argparser(description="make electron density")
    args        = parser.special_parse_args()
    logging.info("Starting.... make_densities")
    logging.info(' '.join(sys.argv))

    pdb_file    = os.path.join(args.main_dir, read_config.get_filename(args.config_file, 'make_densities', "in_pdb_file"))
    aux_dir     = os.path.join(args.main_dir, read_config.get_filename(args.config_file, 'make_densities', "scatt_dir"))
    den_file    = os.path.join(args.main_dir, read_config.get_filename(args.config_file, 'make_densities', "out_density_file"))
    to_write    = py_utils.check_to_overwrite(den_file)

    if to_write:
        timer       = py_utils.my_timer()
        pm          = read_config.get_detector_config(args.config_file, show=args.vb)
        q_pm        = read_config.compute_q_params(pm['detd'], pm['detsize'], pm['pixsize'], pm['wavelength'], show=args.vb)
        timer.reset_and_report("Reading experiment parameters") if args.vb else timer.reset()

        fov_len     = int(np.ceil(q_pm['fov_in_A']/q_pm['half_p_res']) + 1)
        eV          = process_pdb.wavelength_in_A_to_eV(pm['wavelength'])

        atom_types  = process_pdb.find_atom_types(pdb_file)
        scatt_list  = process_pdb.make_scatt_list(atom_types, aux_dir, eV)
        atoms       = process_pdb.get_atom_coords(pdb_file, scatt_list)
        (s_l, t_l)  = process_pdb.read_symmetry(pdb_file)
        all_atoms   = process_pdb.apply_symmetry(atoms, s_l, t_l)
        timer.reset_and_report("Reading PDB") if args.vb else timer.reset()

        den         = process_pdb.atoms_to_density_map(all_atoms, q_pm['half_p_res'])
        lp_den      = process_pdb.low_pass_filter_density_map(den)
        timer.reset_and_report("Creating density map") if args.vb else timer.reset()

        py_utils.write_density(den_file, lp_den, binary=True)
        timer.reset_and_report("Writing densities to file") if args.vb else timer.reset()

        timer.report_time_since_beginning() if args.vb else timer.reset()
