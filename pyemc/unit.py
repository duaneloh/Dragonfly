#!/usr/bin/env python

from __future__ import print_function
import unittest
import numpy.testing as npt

import argparse
import numpy as np
import scipy.special
import os
import sys
import csv
import shutil
from six.moves import configparser
from builtins import bytes
from mpi4py import MPI

import detector
import dataset
import quat
import params
import interp
import iterate
import max_emc

class DragonflyConfig():
    def __init__(self, fname):
        self.config = configparser.RawConfigParser()
        self.fname = fname
        self.config.read(fname.decode('utf-8'))

    def modify_entry(self, section, param, value):
        self.config.set(section, param, value)
        with open(self.fname, 'w') as f:
            self.config.write(f)

    def remove_entry(self, section, param):
        self.config.remove_option(section, param)
        with open(self.fname, 'w') as f:
            self.config.write(f)

class TestDetector(unittest.TestCase):
    def det_sim_tests(self, det, old_style=False, single=True):
        self.assertEqual(det.num_pix, 10201)
        self.assertEqual(det.rel_num_pix, 7540)
        if not old_style:
            self.assertAlmostEqual(det.detd, 585.9375)
            self.assertAlmostEqual(det.ewald_rad, 585.9375)
        self.assertEqual((det.mask==0).sum(), 7540)
        self.assertEqual((det.mask==1).sum(), 2356)
        self.assertEqual((det.mask==2).sum(), 305)
        arr = np.ones(101, dtype='u1')
        arr[41:60] = 0
        npt.assert_equal(det.mask[101:202], arr)
        if single:
            self.assertEqual(det.num_det, 1)
            npt.assert_array_equal(det.mapping, np.zeros(1024, dtype='i4'))

    def test_parse_detector(self):
        print('=== Testing parse_detector() ===')
        det = detector.detector()
        self.assertAlmostEqual(det.parse_detector(recon_folder+b'/data/det_sim.dat'), 70.32817314646061) 
        self.assertEqual(det.num_dfiles, 0)
        self.det_sim_tests(det)
        det.parse_detector(recon_folder+b'/data/det_sim.dat')
        
        det_fname = recon_folder+b'/data/det_sim_test.dat'
        with open(recon_folder+b'/data/det_sim.dat', 'r') as f:
            lines = f.readlines()
        lines[0] = lines[0].split()[0]+'\n'
        with open(det_fname, 'w') as f:
            f.writelines(lines)
        det.parse_detector(det_fname)
        self.det_sim_tests(det, old_style=True)
        os.remove(det_fname)

    def test_parse_detector_list(self):
        print('=== Testing parse_detector_list() ===')
        shutil.copyfile(recon_folder+b'/data/det_sim.dat', recon_folder+b'/data/det_sim_test.dat')
        list_fname = 'test_det_list.txt'
        
        det = detector.detector()
        with open(list_fname, 'w') as f:
            f.writelines([(recon_folder+b'/data/det_sim.dat\n').decode('utf-8'), (recon_folder+b'/data/det_sim.dat\n').decode('utf-8')])
        self.assertAlmostEqual(det.parse_detector_list(bytes(list_fname, 'utf-8')), 70.32817314646061) 
        self.assertEqual(det.num_dfiles, 2)
        self.det_sim_tests(det)
        self.assertIs(det.nth_det(1), None)
        
        det = detector.detector()
        with open(list_fname, 'w') as f:
            f.writelines([(recon_folder+b'/data/det_sim.dat\n').decode('utf-8'), (recon_folder+b'/data/det_sim_test.dat\n').decode('utf-8')])
        det.parse_detector_list(bytes(list_fname, 'utf-8'))
        self.assertEqual(det.num_dfiles, 2)
        self.assertEqual(det.num_det, 2)
        npt.assert_array_equal(det.mapping, [0,1]+1022*[0])
        self.det_sim_tests(det, single=False)
        self.det_sim_tests(det.nth_det(1), single=False)
        self.assertIs(det.nth_det(2), None)
        
        os.remove(list_fname)
        os.remove(recon_folder+b'/data/det_sim_test.dat')

    def test_generate_detectors(self):
        print('=== Testing generate_detectors() ===')
        det = detector.detector()
        self.assertAlmostEqual(det.generate_detectors(config_fname), 70.32817314646061) 
        self.assertEqual(det.num_dfiles, 0)
        self.det_sim_tests(det)
        det.generate_detectors(config_fname)
        
        det = detector.detector()
        list_fname = recon_folder+b'/det_list.txt'
        with open(list_fname, 'w') as f:
            f.writelines(['data/det_sim.dat\n', 'data/det_sim.dat\n'])
        config = DragonflyConfig(config_fname)
        config.modify_entry('emc', 'in_detector_list', 'det_list.txt')
        self.assertRaises(AssertionError, det.generate_detectors, config_fname)
        config.remove_entry('emc', 'in_detector_file')
        det.generate_detectors(config_fname)
        self.det_sim_tests(det)
        
        shutil.copyfile(recon_folder+b'/data/det_sim.dat', recon_folder+b'/data/det_sim_test.dat')
        with open(list_fname, 'w') as f:
            f.writelines(['data/det_sim.dat\n', 'data/det_sim_test.dat\n'])
        det.generate_detectors(config_fname)
        self.det_sim_tests(det, single=False)
        self.det_sim_tests(det.nth_det(1), single=False)
        
        os.remove(list_fname)
        os.remove(recon_folder+b'/data/det_sim_test.dat')
        config.remove_entry('emc', 'in_detector_list')
        config.modify_entry('emc', 'in_detector_file', 'make_detector:::out_detector_file')

    def test_free_detector(self):
        print('=== Testing free_detector() ===')
        det = detector.detector()
        det.free_detector()
        det.free_detector()
        self.assertIsNone(det.num_pix)
        
        det = detector.detector()
        print(config_fname)
        det.generate_detectors(config_fname)
        det.free_detector()
        det.free_detector()
        self.assertIsNone(det.num_pix)

class TestDataset(unittest.TestCase):
    def create_det(self):
        det = detector.detector()
        det.parse_detector(recon_folder+b'/data/det_sim.dat')
        return det

    def photons_tests(self, dset, num_dset=1, first_dset=True):
        # Output of $ ./make_data -T -t 4
        self.assertEqual(dset.num_data, 3000)
        self.assertEqual(dset.num_pix, 10201)
        self.assertEqual(os.path.normpath(dset.filename), os.path.normpath(recon_folder+b'/data/photons.emc'))
        self.assertAlmostEqual(dset.mean_count, 1424.309)
        if first_dset:
            self.assertEqual(dset.tot_num_data, num_dset*3000)
            self.assertAlmostEqual(dset.tot_mean_count, 1424.309)
        if dset.type == 0:
            npt.assert_array_equal(dset.ones_accum, dset.ones.cumsum() - dset.ones)
            npt.assert_array_equal(dset.multi_accum, dset.multi.cumsum() - dset.multi)
            self.assertEqual(dset.ones_total, dset.ones.sum())
            self.assertEqual(dset.multi_total, dset.multi.sum())
            
            npt.assert_array_equal(dset.ones[:3], [541, 426, 429])
            npt.assert_array_equal(dset.ones[-3:], [404, 377, 433])
            npt.assert_array_equal(dset.multi[:3], [384, 326, 226])
            npt.assert_array_equal(dset.multi[-3:], [217, 249, 313])
            npt.assert_array_equal(dset.place_ones[:3], [444, 546, 656])
            npt.assert_array_equal(dset.place_ones[-3:], [8816, 9013, 9274])
            npt.assert_array_equal(dset.place_multi[:3], [1984, 2066, 2161])
            npt.assert_array_equal(dset.place_multi[-3:], [7538, 8015, 8331])
            npt.assert_array_equal(dset.count_multi[11:16], [2,3,2,2,3])
            npt.assert_array_equal(dset.count_multi[-16:-11], [2,2,2,2,3])

    def test_generate_data(self):
        print('=== Testing generate_data() ===')
        det = self.create_det()
        dset = dataset.dataset(det)
        dset.generate_data(config_fname)
        self.photons_tests(dset)
        dset.generate_data(config_fname)
        
        list_fname = recon_folder+b'/test_photons_list.txt'
        with open(list_fname, 'w') as f:
            f.writelines(['data/photons.emc\n', 'data/photons.emc\n'])
        config = DragonflyConfig(config_fname)
        config.modify_entry('emc', 'in_photons_list', list_fname.decode('utf-8'))
        self.assertRaises(AssertionError, dset.generate_data, config_fname)
        config.remove_entry('emc', 'in_photons_file')
        dset.generate_data(config_fname)
        self.photons_tests(dset, 2)
        ndset = dset.next
        self.photons_tests(ndset, 2, False)
        
        os.remove(list_fname)
        config.remove_entry('emc', 'in_photons_list')
        config.modify_entry('emc', 'in_photons_file', 'make_data:::out_photons_file')

    def test_parse_dataset(self):
        print('=== Testing parse_dataset() ===')
        det = self.create_det()
        dset = dataset.dataset(det)
        dset.parse_dataset(recon_folder+b'/data/photons.emc')
        self.photons_tests(dset)
        dset.parse_dataset(recon_folder+b'/data/photons.emc')

    def test_parse_data(self):
        print('=== Testing parse_data() ===')
        det = self.create_det()
        dset = dataset.dataset(det)
        list_fname = b'test_dset_flist.txt'
        with open(list_fname, 'w') as f:
            f.writelines([(recon_folder+b'/data/photons.emc\n').decode('utf-8'), (recon_folder+b'/data/photons.emc\n').decode('utf-8')])
        num_dsets = dset.parse_data(list_fname)
        self.photons_tests(dset, num_dsets)
        ndset = dset.next
        self.photons_tests(ndset, num_dsets, False)
        dset.parse_data(list_fname)
        os.remove(list_fname)

    def test_calc_sum_fact(self):
        print('=== Testing calc_sum_fact() ===')
        det = self.create_det()
        dset = dataset.dataset(det)
        dset.parse_dataset(recon_folder+b'/data/photons.emc')
        dset.calc_sum_fact()
        
        frame = np.zeros(dset.num_pix, dtype='i4')
        frame[dset.place_ones[dset.ones_accum[0]:dset.ones_accum[0]+dset.ones[0]]] = 1
        frame[dset.place_multi[dset.multi_accum[0]:dset.multi_accum[0]+dset.multi[0]]] = dset.count_multi[dset.multi_accum[0]:dset.multi_accum[0]+dset.multi[0]]
        self.assertAlmostEqual(np.log(scipy.special.factorial(frame)).sum(), dset.sum_fact[0])
        
        frame = np.zeros(dset.num_pix, dtype='i4')
        frame[dset.place_ones[dset.ones_accum[-1]:dset.ones_accum[-1]+dset.ones[-1]]] = 1
        frame[dset.place_multi[dset.multi_accum[-1]:dset.multi_accum[-1]+dset.multi[-1]]] = dset.count_multi[dset.multi_accum[-1]:dset.multi_accum[-1]+dset.multi[-1]]
        self.assertAlmostEqual(np.log(scipy.special.factorial(frame)).sum(), dset.sum_fact[-1])

    def test_generate_blacklist(self):
        print('=== Testing generate_blacklist() ===')
        det = self.create_det()
        dset = dataset.dataset(det)
        dset.parse_dataset(recon_folder+b'/data/photons.emc')
        dset.generate_blacklist(config_fname)
        self.assertEqual(dset.blacklist.shape[0], 3000)
        self.assertEqual(dset.blacklist.sum(), 0)
        dset.generate_blacklist(config_fname)
        
        blist_fname = recon_folder+b'/data/blacklist.dat'
        blist = np.zeros(dset.tot_num_data, dtype='u1')
        blist[:10] = 1
        np.savetxt(blist_fname.decode('utf-8'), blist, fmt='%d')
        config = DragonflyConfig(config_fname)
        config.modify_entry('emc', 'blacklist_file', 'data/blacklist.dat')
        dset.generate_blacklist(config_fname)
        self.assertEqual(dset.blacklist.shape[0], 3000)
        self.assertEqual(dset.blacklist.sum(), 10)
        os.remove(blist_fname)
        
        config.remove_entry('emc', 'blacklist_file')
        config.modify_entry('emc', 'selection', 'odd_only')
        dset.generate_blacklist(config_fname)
        self.assertEqual(dset.blacklist.shape[0], 3000)
        self.assertEqual(dset.blacklist.sum(), 1500)
        config.remove_entry('emc', 'selection')

    def test_make_blacklist(self):
        print('=== Testing make_blacklist() ===')
        det = self.create_det()
        dset = dataset.dataset(det)
        dset.parse_dataset(recon_folder+b'/data/photons.emc')
        dset.make_blacklist(b'')
        self.assertEqual(dset.blacklist.shape[0], 3000)
        self.assertEqual(dset.blacklist.sum(), 0)
        
        dset.make_blacklist(b'', odd_flag=2)
        self.assertEqual(dset.blacklist.shape[0], 3000)
        self.assertEqual(dset.blacklist.sum(), 1500)
        npt.assert_array_equal(dset.blacklist[:4], [0,1,0,1])
        
        dset.make_blacklist(b'', odd_flag=1)
        self.assertEqual(dset.blacklist.shape[0], 3000)
        self.assertEqual(dset.blacklist.sum(), 1500)
        npt.assert_array_equal(dset.blacklist[:4], [1,0,1,0])
        
        blist_fname = recon_folder+b'/data/blacklist.dat'
        blist = np.zeros(dset.tot_num_data, dtype='u1')
        blist[:10] = 1
        np.savetxt(blist_fname.decode('utf-8'), blist, fmt='%d')
        dset.make_blacklist(blist_fname)
        self.assertEqual(dset.blacklist.shape[0], 3000)
        self.assertEqual(dset.blacklist.sum(), 10)
        npt.assert_array_equal(dset.blacklist[8:12], [1,1,0,0])
        
        # Behavior when both blacklist file and odd/even selection
        # Alternate frames which are not blacklisted by file are blacklisted
        dset.make_blacklist(blist_fname, odd_flag=2)
        self.assertEqual(dset.blacklist.shape[0], 3000)
        self.assertEqual(dset.blacklist.sum(), 1505)
        npt.assert_array_equal(dset.blacklist[8:12], [1,1,0,1])
        
        dset.make_blacklist(blist_fname, odd_flag=1)
        self.assertEqual(dset.blacklist.shape[0], 3000)
        self.assertEqual(dset.blacklist.sum(), 1505)
        npt.assert_array_equal(dset.blacklist[8:12], [1,1,1,0])
        os.remove(blist_fname)

    def test_free_data(self):
        print('=== Testing free_data() ===')
        det = self.create_det()
        dset = dataset.dataset(det)
        dset.parse_dataset(recon_folder+b'/data/photons.emc')
        dset.free_data()
        dset.free_data()
        self.assertIsNone(dset.num_data)

class TestRotation(unittest.TestCase):
    def quat_tests(self, rot):
        self.assertEqual(rot.num_rot, 3240)
        self.assertFalse(rot.icosahedral_flag)
        self.assertEqual(rot.quat.shape, (3240, 5))
        npt.assert_array_almost_equal(rot.quat[0], [0.5,-0.5,-0.5,-0.5, 2.07312814e-04])
        npt.assert_array_almost_equal(rot.quat[-1], [2.18508012e-01, 0.00000000e+00, 5.72061403e-01, 7.90569415e-01, 3.38911452e-04])

    def test_generate_quaternion(self):
        print('=== Testing generate_quaternion() ===')
        rot = quat.rotation()
        rot.generate_quaternion(config_fname)
        self.quat_tests(rot)
        rot.generate_quaternion(config_fname)
        quat_fname = recon_folder+b'/data/quat.dat'
        with open(quat_fname, 'w') as f:
            w = csv.writer(f, delimiter=' ')
            w.writerow([rot.num_rot])
            w.writerows(rot.quat)
        
        config = DragonflyConfig(config_fname)
        config.modify_entry('emc', 'sym_icosahedral', '1')
        rot.generate_quaternion(config_fname)
        self.assertTrue(rot.icosahedral_flag)
        self.assertEqual(rot.num_rot, 75)
        npt.assert_array_almost_equal(rot.quat[0], [1., 0., 0., 0., 0.00881278])
        npt.assert_array_almost_equal(rot.quat[1], [0.98830208, -0.12973191, -0.08017873,  0., 0.01192908])
        config.remove_entry('emc', 'sym_icosahedral')
        
        rot.free_quat()
        rot = quat.rotation()
        config.modify_entry('emc', 'in_quat_file', 'data/quat.dat')
        self.assertRaises(AssertionError, rot.generate_quaternion, config_fname)
        config.remove_entry('emc', 'num_div')
        rot.generate_quaternion(config_fname)
        self.quat_tests(rot)
        
        os.remove(quat_fname)
        config.modify_entry('emc', 'num_div', '4')
        config.remove_entry('emc', 'in_quat_file')

    def test_quat_gen(self):
        print('=== Testing quat_gen() ===')
        rot = quat.rotation()
        self.assertEqual(rot.quat_gen(4), 3240)
        self.quat_tests(rot)
        for i in range(1,13,1):
            self.assertEqual(rot.quat_gen(i), 10*(5*i**3 + i))

    def test_parse_quat(self):
        print('=== Testing parse_quat() ===')
        rot = quat.rotation()
        self.assertEqual(rot.parse_quat(b''), -1)
        
        rot.quat_gen(4)
        quat_fname = recon_folder+b'/data/quat.dat'
        with open(quat_fname, 'w') as f:
            w = csv.writer(f, delimiter=' ')
            w.writerow([rot.num_rot])
            w.writerows(rot.quat)
        self.assertEqual(rot.parse_quat(quat_fname), 3240)
        self.quat_tests(rot)
        os.remove(quat_fname)

    def test_divide_quat(self):
        print('=== Testing divide_quat() ===')
        rot = quat.rotation()
        rot.quat_gen(4)
        rot.divide_quat(0, 1, 1)
        self.assertEqual(rot.num_rot_p, 3240)
        rot.divide_quat(5, 7, 1)
        self.assertEqual(rot.num_rot_p, 463)
        rot.divide_quat(6, 7, 1)
        self.assertEqual(rot.num_rot_p, 462)

    def test_free_quat(self):
        print('=== Testing free_quat() ===')
        rot = quat.rotation()
        rot.quat_gen(4)
        rot.divide_quat(6, 7, 1)
        rot.free_quat()
        rot.free_quat()
        self.assertIsNone(rot.num_rot)

class TestParams(unittest.TestCase):
    def configparams_test(self, param, default=True):
        self.assertEqual(param.rank, 0)
        self.assertEqual(param.num_proc, 1)
        self.assertEqual(param.known_scale, 0)
        self.assertEqual(param.start_iter, 1)
        if default:
            self.assertEqual(os.path.abspath(param.output_folder), os.path.abspath(recon_folder+b'/data/'))
            self.assertEqual(os.path.abspath(param.log_fname), os.path.abspath(recon_folder+b'/EMC.log'))
            self.assertEqual(param.need_scaling, 0)
            self.assertEqual(param.alpha, 0.)
            self.assertEqual(param.beta, 1.)
            self.assertEqual(param.beta_period, 100)
            self.assertEqual(param.beta_jump, 1.)
            self.assertEqual(param.sigmasq, 0.)
        else:
            self.assertEqual(os.path.abspath(param.output_folder), os.path.abspath(recon_folder+b'/other_data/'))
            self.assertEqual(os.path.abspath(param.log_fname), os.path.abspath(recon_folder+b'/other_EMC.log'))
            self.assertEqual(param.need_scaling, 1)
            self.assertEqual(param.alpha, 0.5)
            self.assertEqual(param.beta, 0.5)
            self.assertEqual(param.beta_period, 10)
            self.assertEqual(param.beta_jump, 1.5)
            self.assertEqual(param.sigmasq, 1.)
        
    def test_generate_params(self):
        print('=== Testing generate_params() ===')
        param = params.params()
        param.generate_params(config_fname)
        self.configparams_test(param)
        param.generate_params(config_fname)
        
        config = DragonflyConfig(config_fname)
        config.modify_entry('emc', 'output_folder', 'other_data/')
        config.modify_entry('emc', 'log_file', 'other_EMC.log')
        config.modify_entry('emc', 'need_scaling', '1')
        config.modify_entry('emc', 'alpha', '0.5')
        config.modify_entry('emc', 'beta', '0.5')
        config.modify_entry('emc', 'beta_schedule', '1.5 10')
        config.modify_entry('emc', 'gaussian_sigma', '1.')
        param.generate_params(config_fname)
        self.configparams_test(param, default=False)
        
        config.modify_entry('emc', 'output_folder', 'data/')
        config.modify_entry('emc', 'log_file', 'EMC.log')
        config.remove_entry('emc', 'need_scaling')
        config.remove_entry('emc', 'alpha')
        config.remove_entry('emc', 'beta')
        config.remove_entry('emc', 'beta_schedule')
        config.remove_entry('emc', 'gaussian_sigma')

    def test_generate_output_dirs(self):
        print('=== Testing generate_output_dirs() ===')
        param = params.params()
        param.generate_params(config_fname)
        flist = [recon_folder+b'/data/'+d for d in [b'output', b'weights', b'orientations', b'scale', b'likelihood', b'mutualInfo']]
        [shutil.rmtree(d) for d in flist if os.path.exists(d)]
        param.generate_output_dirs()
        self.assertTrue(np.array([os.path.exists(d) for d in flist]).all())
        param.generate_output_dirs()

    def test_free_params(self):
        print('=== Testing free_params() ===')
        param = params.params()
        param.generate_params(config_fname)
        param.free_params()
        param.free_params()

class TestInterp(unittest.TestCase):
    def test_make_rot_quat(self):
        print('=== Testing make_rot_quat() ===')
        npt.assert_array_almost_equal(interp.make_rot_quat(np.array([1,0,0,0], dtype='f8')), np.identity(3))
        npt.assert_array_almost_equal(interp.make_rot_quat(0.5*np.array([1,-1,-1,-1], dtype='f8')), [[0,0,1],[1,0,0],[0,1,0]])
        npt.assert_array_almost_equal(interp.make_rot_quat(0.5*np.array([1,-1,-1,+1], dtype='f8')), [[0,1,0],[0,0,-1],[-1,0,0]])
        npt.assert_array_almost_equal(interp.make_rot_quat(0.5*np.array([1,+1,+1,-1], dtype='f8')), [[0,0,-1],[1,0,0],[0,-1,0]])
        npt.assert_array_almost_equal(interp.make_rot_quat(np.array([0,0,1/np.sqrt(2.),-1/np.sqrt(2.)], dtype='f8')), [[-1,0,0],[0,0,-1],[0,-1,0]])
        npt.assert_array_almost_equal(interp.make_rot_quat(np.array([1/np.sqrt(2.),0,1/np.sqrt(2.),0], dtype='f8')), [[0,0,-1],[0,1,0],[1,0,0]])
        npt.assert_array_almost_equal(interp.make_rot_quat(np.array([-1/np.sqrt(2.),1/np.sqrt(2.),0,0], dtype='f8')), [[1,0,0],[0,0,-1],[0,1,0]])

    def test_symmetrize_friedel(self):
        print('=== Testing symmetrize_friedel() ===')
        arr = np.arange(27.).reshape(3,3,3)
        interp.symmetrize_friedel(arr)
        npt.assert_array_almost_equal(arr, 13.*np.ones((3,3,3), dtype='f8'))
        arr = np.zeros((3,3,3), dtype='f8'); arr[0] = 2.2
        interp.symmetrize_friedel(arr)
        npt.assert_array_almost_equal(arr, np.concatenate((1.1*np.ones(9), np.zeros(9), 1.1*np.ones(9))).reshape(3,3,3))
        arr = np.zeros((3,3,3), dtype='f8'); arr[:,0] = 2.2
        interp.symmetrize_friedel(arr)
        npt.assert_array_almost_equal(arr, np.concatenate((1.1*np.ones(9), np.zeros(9), 1.1*np.ones(9))).reshape(3,3,3).transpose(1,0,2))
        arr = np.zeros((3,3,3), dtype='f8'); arr[:,:,0] = 2.2
        interp.symmetrize_friedel(arr)
        npt.assert_array_almost_equal(arr, np.concatenate((1.1*np.ones(9), np.zeros(9), 1.1*np.ones(9))).reshape(3,3,3).transpose(2,1,0))

    def test_rotate_model(self):
        print('=== Testing rotate_model() ===')
        model = np.random.random((101,101,101))
        rotmodel = np.zeros_like(model)
        interp.rotate_model(np.identity(3), model, rotmodel)
        self.assertAlmostEqual(np.linalg.norm((model-rotmodel)[1:-1,1:-1,1:-1].flatten()), 0.)
        rotmodel.fill(0.)
        interp.rotate_model(np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype='f8'), model, rotmodel)
        self.assertAlmostEqual(np.linalg.norm((np.rot90(model,1,axes=(2,1))-rotmodel)[1:-1,1:-1,1:-1].flatten()), 0.)
        rotmodel.fill(0.)
        interp.rotate_model(np.array([[0,-1,0],[1,0,0],[0,0,1]], dtype='f8'), model, rotmodel)
        self.assertAlmostEqual(np.linalg.norm((np.rot90(model,1,axes=(1,0))-rotmodel)[1:-1,1:-1,1:-1].flatten()), 0.)
        
        intens = 1.e-9*np.fromfile(recon_folder+b'/data/intensities.bin').reshape(3*(145,))
        quat = np.array([np.sqrt(0.86),0.1,0.2,0.3])
        rot = interp.make_rot_quat(quat)
        rotmodel = np.zeros_like(intens)
        interp.rotate_model(rot, intens, rotmodel)
        npt.assert_array_almost_equal(rotmodel[100:103,100:103,100:103], [[[0.09167898, 0.03647299, 0.00707748], [0.12777323, 0.06260195, 0.02031047], [0.16907458, 0.09657051, 0.04333556]], [[0.06231541, 0.02152542, 0.00546131], [0.09500319, 0.04364314, 0.0141322], [0.13197364, 0.07370418, 0.03259982]], [[0.04283407, 0.01201542, 0.00398634], [0.06886472, 0.02839342, 0.00625303], [0.10017888, 0.05289991, 0.02144472]]])

    def test_slice_gen3d(self):
        print('=== Testing slice_gen3d() ===')
        det = detector.detector()
        det.parse_detector(recon_folder+b'/data/det_sim.dat')
        intens = 1.e-9*np.fromfile(recon_folder+b'/data/intensities.bin').reshape(3*(145,))
        view = np.zeros(det.num_pix)
        
        quat = np.array([1.,0,0,0])
        interp.slice_gen3d(quat, view, intens, det)
        self.assertAlmostEqual(view.mean(), 223.922218946)
        npt.assert_array_almost_equal(view[:5], [0.03021473, 0.02554173, 0.01861631, 0.01085438, 0.00459315])
        interp.slice_gen3d(quat, view, intens, det, rescale=1.)
        self.assertAlmostEqual(view.mean(), 1.86882056344)
        npt.assert_array_almost_equal(view[:5], [-3.49942586, -3.66744152, -3.98371703, -4.5231864, -5.38318919])
        
        quat = np.array([np.sqrt(0.86),0.1,0.2,0.3])
        interp.slice_gen3d(quat, view, intens, det)
        self.assertAlmostEqual(view.mean(), 184.449773553)
        npt.assert_array_almost_equal(view[:5], [0.00039123, 0.00014522, 0.00057308, 0.00185642, 0.00371838])
        interp.slice_gen3d(quat, view, intens, det, rescale=1.)
        self.assertAlmostEqual(view.mean(), 0.567310517859)
        npt.assert_array_almost_equal(view[:5], [-7.84620536, -8.83729446, -7.46449246, -6.28910363, -5.59446611])

    def test_slice_merge3d(self):
        print('=== Testing slice_merge3d() ===')
        det = detector.detector()
        det.parse_detector(recon_folder+b'/data/det_sim.dat')
        view = np.ascontiguousarray(det.pixels[:,3])
        quat = np.array([np.sqrt(0.86),0.1,0.2,0.3])
        model = np.zeros(3*(145,))
        weight = np.zeros_like(model)
        interp.slice_merge3d(quat, view, model, weight, det)
        npt.assert_array_almost_equal(model, weight)
        npt.assert_array_almost_equal(model[103:106,68:71,82:85], [[[0.05970267, 0.86777407, 0.08261854], [0.29557584, 0.6624112, 0.00868513], [0.69197243, 0.40168333, 0.]], [[0., 0.69936496, 0.34398347], [0.07919628, 1.25519294, 0.0490487], [0.38575382, 0.65815609, 0.]], [[0., 0.45319003, 0.65207203], [0.01374454, 0.68324196, 0.25491581], [0.12366799, 0.84800088, 0.05462553]]])
        
        view = np.ascontiguousarray(det.pixels[:,3])*np.arange(det.num_pix)
        model = np.zeros(3*(145,))
        weight2 = np.zeros_like(model)
        interp.slice_merge3d(quat, view, model, weight2, det)
        npt.assert_array_almost_equal(weight, weight2)
        npt.assert_array_almost_equal(model[103:106,68:71,82:85], [[[480.23952805, 7053.79230354, 672.87605846], [2377.76518912, 5341.74335349, 70.74035788], [5519.30333337, 3220.70729727, 0.]], [[0., 5738.38868753, 2835.8821622], [640.53422865, 10226.00092727, 403.93064498], [3106.35276845, 5325.18474032, 0.]], [[0., 3752.37021246, 5424.77431879], [111.97675292, 5624.55664759, 2102.41717762], [1007.59124681, 6925.69283809, 450.55910447]]])

    def test_slice_gen2d(self):
        print('=== Testing slice_gen2d() ===')
        det = detector.detector()
        det.parse_detector(recon_folder+b'/data/det_sim.dat', norm_flag=-3)
        intens = np.arange(145*145*3).astype('f8').reshape(3,145,145)
        view = np.zeros(det.num_pix)
        
        angle = np.array([1.37*np.pi])
        interp.slice_gen2d(angle, view, intens, det)
        self.assertAlmostEqual(view.mean(), 10512.)
        npt.assert_array_almost_equal(view[:5], [6674.995665, 6808.058374, 6941.174136, 7074.339593, 7207.551378])
        interp.slice_gen2d(angle, view, intens, det, rescale=1.)
        self.assertAlmostEqual(view.mean(), 9.1579227696454524)
        npt.assert_array_almost_equal(view[:5], [8.806124, 8.825862, 8.845226, 8.864229, 8.882885])
        
        angle = np.array([3.14*np.pi])
        interp.slice_gen2d(angle, view, intens[1:], det)
        self.assertAlmostEqual(view.mean(), 31537.)
        npt.assert_array_almost_equal(view[:5], [34414.902109, 34489.21956, 34563.301629, 34637.146226, 34710.751266])
        interp.slice_gen2d(angle, view, intens[1:], det, rescale=1.)
        self.assertAlmostEqual(view.mean(), 10.349731872416205)
        npt.assert_array_almost_equal(view[:5], [10.446245, 10.448402, 10.450548, 10.452682, 10.454805])
        
    def test_slice_merge2d(self):
        print('=== Testing slice_merge2d() ===')
        det = detector.detector()
        qmax = det.parse_detector(recon_folder+b'/data/det_sim.dat', norm_flag=-3)
        view = np.ascontiguousarray(det.pixels[:,2])
        angle = np.array([1.37*np.pi])
        model = np.zeros((3, 145, 145))
        weight = np.zeros_like(model)
        interp.slice_merge2d(angle, view, model, weight, det)
        npt.assert_array_almost_equal(model, weight)
        npt.assert_array_almost_equal(model[0,88:91,82:85], [[1.013543, 1.013526, 0.95921],[0.948738, 0.988016, 0.986792], [1.013526, 0.985034, 0.98392]])
        
        view = np.ascontiguousarray(det.pixels[:,2])*np.arange(det.num_pix)
        model = np.zeros((3, 145, 145))
        weight2 = np.zeros_like(model)
        interp.slice_merge2d(angle, view, model, weight2, det)
        npt.assert_array_almost_equal(weight, weight2)
        npt.assert_array_almost_equal(model[0,88:91,82:85], [[3590.950843, 3495.003523, 3221.593928], [3314.725198, 3367.85047, 3269.304046], [3513.177067, 3323.48564, 3226.194119]])

class TestIterate(unittest.TestCase):
    def allocate_iterate(self):
        itr = iterate.iterate()
        det = detector.detector()
        dset = dataset.dataset(det)
        param = params.params()
        qmax = det.generate_detectors(config_fname)
        dset.generate_data(config_fname)
        param.generate_params(config_fname)
        dset.generate_blacklist(config_fname)
        itr.generate_iterate(config_fname, qmax, param, det, dset)
        return itr, det, dset, param, qmax

    def test_calculate_size(self):
        print('=== Testing calculate_size() ===')
        itr = iterate.iterate()
        self.assertEqual(itr.calculate_size(12.5), 29)
        self.assertEqual(itr.calculate_size(125.), 29)
        itr.free_iterate()
        itr = iterate.iterate()
        self.assertEqual(itr.calculate_size(12.2), 29)
        itr.free_iterate()
        itr = iterate.iterate()
        self.assertEqual(itr.calculate_size(12.7), 29)
        itr.free_iterate()
        itr = iterate.iterate()
        self.assertEqual(itr.calculate_size(12), 27)
        itr.free_iterate()
        itr = iterate.iterate()
        self.assertEqual(itr.calculate_size(13.), 29)
        itr.free_iterate()

    def test_generate_iterate(self):
        print('=== Testing generate_iterate() ===')
        itr, det, dset, param, qmax = self.allocate_iterate()
        self.assertEqual(itr.size, 145)
        self.assertRaises(AssertionError, itr.generate_iterate, config_fname, qmax, param, det, dset, continue_flag=True)
        itr.generate_iterate(config_fname, qmax, param, det, dset, config_section=b'foobar')
        
        config = DragonflyConfig(config_fname)
        config.modify_entry('emc', 'size', '101')
        config.modify_entry('emc', 'need_scaling', '1')
        itr, det, dset, param, qmax = self.allocate_iterate()
        npt.assert_array_equal(itr.scale, np.ones(dset.tot_num_data, dtype='f8'))
        npt.assert_array_equal(dset.count[:5], [1621, 1382, 1050, 2436, 1450])
        npt.assert_array_equal(dset.count[-5:], [1093, 1597, 1053, 1080, 1315])
        self.assertEqual(itr.size, 101)
        
        config.remove_entry('emc', 'size')
        config.remove_entry('emc', 'need_scaling')

    def test_calc_scale(self):
        print('=== Testing calc_scale() ===')
        itr, det, dset, param, qmax = self.allocate_iterate()
        itr.calc_scale(dset, det)
        self.assertEqual(itr.scale.shape[0], dset.tot_num_data)
        npt.assert_array_equal(itr.scale, np.ones(dset.tot_num_data, dtype='f8'))
        npt.assert_array_equal(dset.count[:5], [1621, 1382, 1050, 2436, 1450])
        npt.assert_array_equal(dset.count[-5:], [1093, 1597, 1053, 1080, 1315])
        itr.calc_scale(dset, det, print_fname=recon_folder+b'/data/scale/scale_000.dat')

    def test_normalize_scale(self):
        print('=== Testing normalize_scale() ===')
        itr, det, dset, param, qmax = self.allocate_iterate()
        itr.calc_scale(dset, det)
        itr.normalize_scale(dset)
        config = DragonflyConfig(config_fname)
        config.modify_entry('emc', 'need_scaling', '1')
        itr, det, dset, param, qmax = self.allocate_iterate()
        itr.normalize_scale(dset)
        config.remove_entry('emc', 'need_scaling')

    def test_parse_scale(self):
        print('=== Testing parse_scale() ===')
        itr, det, dset, param, qmax = self.allocate_iterate()
        itr.calc_scale(dset, det)
        self.assertEqual(itr.parse_scale(b''), 0)
        
        scale_fname = recon_folder+b'/data/scales.dat'
        rand_scales = np.random.random(dset.tot_num_data)
        np.savetxt(scale_fname.decode('utf-8'), rand_scales)
        self.assertEqual(itr.parse_scale(scale_fname), 1)
        npt.assert_array_almost_equal(itr.scale, rand_scales)

    def test_parse_input(self):
        print('=== Testing parse_input() ===')
        itr, det, dset, param, qmax = self.allocate_iterate()
        print(itr.modes)
        itr.parse_input(b'', -1.)
        self.assertAlmostEqual(itr.model1.mean(), 0.499965209377)

    def test_free_iterate(self):
        print('=== Testing free_iterate() ===')
        itr = iterate.iterate()
        itr.free_iterate()
        itr.free_iterate()
        itr, det, dset, param, qmax = self.allocate_iterate()
        itr.free_iterate()
        itr.free_iterate()

class TestMaxEMC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('=== Initializing maximize() ===')
        cls.maximize = max_emc.py_maximize(config_fname, quiet_setup=False)

    def test_allocate_memory(self):
        print('=== Testing static allocate_memory() ===')
        zarr = np.zeros(3000)
        zvol = np.zeros(3*(145,))
        
        data = max_emc.py_max_data(within_openmp=False)
        self.maximize.allocate_memory(data)
        npt.assert_array_equal(data.max_exp, zarr)
        npt.assert_array_equal(data.p_sum, zarr)
        npt.assert_array_equal(data.info, zarr)
        npt.assert_array_equal(data.likelihood, zarr)
        npt.assert_array_equal(data.rmax, np.zeros(3000, dtype='i4'))
        npt.assert_array_equal(data.u, np.zeros(3240))
        npt.assert_array_equal(data.max_exp_p, -np.ones(3000)*sys.float_info.max)
        self.assertEqual(data.probab.shape, (3240,3000))
        self.maximize.free_memory(data)
        
        data = max_emc.py_max_data(within_openmp=True)
        self.maximize.allocate_memory(data)
        npt.assert_array_equal(data.info, zarr)
        npt.assert_array_equal(data.likelihood, zarr)
        npt.assert_array_equal(data.p_sum, np.zeros(1))
        npt.assert_array_equal(data.rmax, np.zeros(3000, dtype='i4'))
        npt.assert_array_equal(data.max_exp_p, -np.ones(3000)*sys.float_info.max)
        npt.assert_array_equal(data.model, zvol)
        npt.assert_array_equal(data.weight, zvol)
        self.assertEqual(len(data.all_views), 1)
        self.assertEqual(len(data.all_views[0]), 10201)
        self.maximize.free_memory(data)
        
    def test_calculate_rescale(self):
        print('=== Testing static calculate_rescale() ===')
        data = max_emc.py_max_data(within_openmp=False)
        self.maximize.allocate_memory(data)
        rescale = self.maximize.calculate_rescale(data)
        self.assertAlmostEqual(rescale, 0.9972006898)
        self.maximize.free_memory(data)

    def test_calculate_prob(self):
        print('=== Testing static calculate_prob() ===')
        common_data = max_emc.py_max_data(within_openmp=False)
        self.maximize.allocate_memory(common_data)
        rescale = self.maximize.calculate_rescale(common_data)
        priv_data = max_emc.py_max_data(within_openmp=True)
        self.maximize.allocate_memory(priv_data)
        self.maximize.calculate_prob(0, priv_data, common_data)
        npt.assert_array_almost_equal(common_data.probab[0,:5], [-4300.60968803, -3913.03874035, -3341.71258934, -5798.14606903, -4048.21180691])
        self.maximize.calculate_prob(1, priv_data, common_data)
        npt.assert_array_equal(np.where(priv_data.rmax==1)[0][:5], [1, 2, 6, 9, 13])
        self.maximize.free_memory(common_data)
        self.maximize.free_memory(priv_data)

    def test_normalize_prob(self):
        print('=== Testing static normalize_prob() ===')
        common_data = max_emc.py_max_data(within_openmp=False)
        self.maximize.allocate_memory(common_data)
        priv_data = max_emc.py_max_data(within_openmp=True)
        self.maximize.allocate_memory(priv_data)
        self.maximize.normalize_prob(priv_data, common_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unit testing Dragonfly')
    parser.add_argument('-f', '--recon_folder', help='Reconstruction folder with test data', default=os.path.relpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../testing_0001/')))
    parser.add_argument('unittest_args', nargs='*')
    args = parser.parse_args()

    recon_folder = bytes(args.recon_folder, 'utf-8')
    config_fname = recon_folder + b'/config.ini'
    print(config_fname)
    print('Testing using recon folder: %s'%recon_folder)
    sys.argv[1:] = args.unittest_args
    
    unittest.main(verbosity=0)
