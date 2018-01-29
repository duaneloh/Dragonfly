#!/usr/bin/env python

import unittest
import numpy.testing as npt
import numpy as np
import scipy.special
import os
import sys
import detector
import dataset

recon_folder = '../recon_0001/'

class TestDetector(unittest.TestCase):
    def det_sim_tests(self, det):
        self.assertEqual(det.num_pix, 10201)
        self.assertEqual(det.rel_num_pix, 7540)
        self.assertEqual(det.num_det, 1)
        self.assertAlmostEqual(det.detd, 585.9375)
        self.assertAlmostEqual(det.ewald_rad, 585.9375)
        self.assertEqual((det.mask==0).sum(), 7540)
        self.assertEqual((det.mask==1).sum(), 2356)
        self.assertEqual((det.mask==2).sum(), 305)
        arr = np.ones(101, dtype='u1')
        arr[41:60] = 0
        npt.assert_equal(det.mask[101:202], arr)
        npt.assert_equal(det.mapping, np.zeros(1024, dtype='i4'))
    
    def test_parse_detector(self):
        print('=== Testing parse_detector()')
        det = detector.detector()
        self.assertAlmostEqual(det.parse_detector(recon_folder+'/data/det_sim.dat'), 70.32817314646061) 
        self.assertEqual(det.num_dfiles, 0)
        self.det_sim_tests(det)
        det.parse_detector(recon_folder+'/data/det_sim.dat')

    def test_parse_detector_list(self):
        print('=== Testing parse_detector_list()')
        det = detector.detector()
        list_fname = 'test_det_list.txt'
        with open(list_fname, 'w') as f:
            f.writelines([recon_folder+'/data/det_sim.dat\n', recon_folder+'/data/det_sim.dat\n'])
        self.assertAlmostEqual(det.parse_detector_list(list_fname), 70.32817314646061) 
        self.assertEqual(det.num_dfiles, 2)
        self.det_sim_tests(det)
        det.parse_detector_list(list_fname)
        os.remove(list_fname)

    def test_generate_detectors(self):
        print('=== Testing generate_detectors()')
        det = detector.detector()
        self.assertAlmostEqual(det.generate_detectors(recon_folder+'/config.ini'), 70.32817314646061) 
        self.assertEqual(det.num_dfiles, 0)
        self.det_sim_tests(det)
        det.generate_detectors(recon_folder+'/config.ini')

    def test_free_detector(self):
        print('=== Testing free_detector()')
        det = detector.detector()
        det.free_detector()
        det.free_detector()
        self.assertIsNone(det.num_pix)

class TestDataset(unittest.TestCase):
    def create_det(self):
        det = detector.detector()
        det.parse_detector(recon_folder+'/data/det_sim.dat')
        return det

    def photons_tests(self, dset, num_dset=1, first_dset=True):
        self.assertEqual(dset.num_data, 3000)
        self.assertEqual(dset.num_pix, 10201)
        self.assertEqual(os.path.normpath(dset.filename), os.path.normpath(recon_folder+'/data/photons.emc'))
        self.assertAlmostEqual(dset.mean_count, 1422.04766666)
        if first_dset:
            self.assertEqual(dset.tot_num_data, num_dset*3000)
            self.assertAlmostEqual(dset.tot_mean_count, 1422.04766666)
        if dset.type == 0:
            npt.assert_array_equal(dset.ones_accum, dset.ones.cumsum() - dset.ones)
            npt.assert_array_equal(dset.multi_accum, dset.multi.cumsum() - dset.multi)
            self.assertEqual(dset.ones_total, dset.ones.sum())
            self.assertEqual(dset.multi_total, dset.multi.sum())
            
            npt.assert_array_equal(dset.ones[:3], np.array([541, 426, 429], dtype='i4'))
            npt.assert_array_equal(dset.ones[-3:], np.array([378, 402, 412], dtype='i4'))
            npt.assert_array_equal(dset.multi[:3], np.array([384, 326, 226], dtype='i4'))
            npt.assert_array_equal(dset.multi[-3:], np.array([307, 231, 302], dtype='i4'))
            npt.assert_array_equal(dset.place_ones[:3], np.array([444, 546, 656], dtype='i4'))
            npt.assert_array_equal(dset.place_ones[-3:], np.array([8129, 8214, 8234], dtype='i4'))
            npt.assert_array_equal(dset.place_multi[:3], np.array([1984, 2066, 2161], dtype='i4'))
            npt.assert_array_equal(dset.place_multi[-3:], np.array([7443, 7534, 7836], dtype='i4'))
            npt.assert_array_equal(dset.count_multi[11:16], np.array([2,3,2,2,3], dtype='i4'))
            npt.assert_array_equal(dset.count_multi[-16:-11], np.array([2,3,2,2,2], dtype='i4'))
    
    def test_generate_data(self):
        print('=== Testing parse_dataset()')
        det = self.create_det()
        dset = dataset.dataset(det)
        dset.generate_data(recon_folder+'/config.ini')
        self.photons_tests(dset)
        dset.generate_data(recon_folder+'/config.ini')

    def test_parse_dataset(self):
        print('=== Testing parse_dataset()')
        det = self.create_det()
        dset = dataset.dataset(det)
        dset.parse_dataset(recon_folder+'/data/photons.emc')
        self.photons_tests(dset)
        dset.parse_dataset(recon_folder+'/data/photons.emc')

    def test_parse_data(self):
        print('=== Testing parse_data()')
        det = self.create_det()
        dset = dataset.dataset(det)
        temp_fname = 'test_dset_flist.txt'
        with open(temp_fname, 'w') as f:
            f.writelines([recon_folder+'/data/photons.emc\n', recon_folder+'/data/photons.emc\n'])
        num_dsets = dset.parse_data(temp_fname)
        self.photons_tests(dset, num_dsets)
        ndset = dset.next
        self.photons_tests(ndset, num_dsets, False)
        dset.parse_data(temp_fname)
        os.remove(temp_fname)

    def test_calc_sum_fact(self):
        print('=== Testing calc_sum_fact()')
        det = self.create_det()
        dset = dataset.dataset(det)
        dset.parse_dataset(recon_folder+'/data/photons.emc')
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
        print('=== Testing generate_blacklist()')
        det = self.create_det()
        dset = dataset.dataset(det)
        dset.parse_dataset(recon_folder+'/data/photons.emc')
        dset.generate_blacklist(recon_folder+'/config.ini')
        self.assertEqual(dset.blacklist.shape[0], 3000)
        self.assertEqual(dset.blacklist.sum(), 0)
        dset.generate_blacklist(recon_folder+'/config.ini')

    def test_make_blacklist(self):
        print('=== Testing make_blacklist()')
        det = self.create_det()
        dset = dataset.dataset(det)
        dset.parse_dataset(recon_folder+'/data/photons.emc')
        dset.make_blacklist('')
        self.assertEqual(dset.blacklist.shape[0], 3000)
        self.assertEqual(dset.blacklist.sum(), 0)
        dset.make_blacklist('', odd_flag=2)
        self.assertEqual(dset.blacklist.shape[0], 3000)
        self.assertEqual(dset.blacklist.sum(), 1500)
        npt.assert_array_equal(dset.blacklist[:4], np.array([0,1,0,1], dtype='u1'))
        dset.make_blacklist('', odd_flag=1)
        self.assertEqual(dset.blacklist.shape[0], 3000)
        self.assertEqual(dset.blacklist.sum(), 1500)
        npt.assert_array_equal(dset.blacklist[:4], np.array([1,0,1,0], dtype='u1'))

    def test_free_data(self):
        print('=== Testing make_blacklist()')
        det = self.create_det()
        dset = dataset.dataset(det)
        dset.parse_dataset(recon_folder+'/data/photons.emc')
        dset.free_data()
        dset.free_data()
        self.assertIsNone(dset.num_data)

if __name__ == '__main__':
    unittest.main(verbosity=0)

