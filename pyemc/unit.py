#!/usr/bin/env python

import unittest
import numpy.testing as npt
import numpy as np
import scipy.special
import os
import sys
import shutil
import detector
import dataset
import quat
import params
import interp

recon_folder = os.path.relpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test_0001/'))
# TODO Create function to add/modify config file entries

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
        # TODO Test with old style detector file as well

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
        # Output of $ ./make_data -T -t 4
        self.assertEqual(dset.num_data, 3000)
        self.assertEqual(dset.num_pix, 10201)
        self.assertEqual(os.path.normpath(dset.filename), os.path.normpath(recon_folder+'/data/photons.emc'))
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
        print('=== Testing generate_data()')
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
        dset.make_blacklist('') # TODO Create blacklist file and test
        self.assertEqual(dset.blacklist.shape[0], 3000)
        self.assertEqual(dset.blacklist.sum(), 0)
        dset.make_blacklist('', odd_flag=2)
        self.assertEqual(dset.blacklist.shape[0], 3000)
        self.assertEqual(dset.blacklist.sum(), 1500)
        npt.assert_array_equal(dset.blacklist[:4], [0,1,0,1])
        dset.make_blacklist('', odd_flag=1)
        self.assertEqual(dset.blacklist.shape[0], 3000)
        self.assertEqual(dset.blacklist.sum(), 1500)
        npt.assert_array_equal(dset.blacklist[:4], [1,0,1,0])

    def test_free_data(self):
        print('=== Testing free_data()')
        det = self.create_det()
        dset = dataset.dataset(det)
        dset.parse_dataset(recon_folder+'/data/photons.emc')
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
        print('=== Testing generate_quaternion()')
        rot = quat.rotation()
        rot.generate_quaternion(recon_folder+'/config.ini')
        self.quat_tests(rot)
        rot.generate_quaternion(recon_folder+'/config.ini')
        # TODO Modify config to test icosahedral reduction

    def test_quat_gen(self):
        print('=== Testing quat_gen()')
        rot = quat.rotation()
        self.assertEqual(rot.quat_gen(4), 3240)
        self.quat_tests(rot)
        for i in range(1,13,1):
            self.assertEqual(rot.quat_gen(i), 10*(5*i**3 + i))

    def test_parse_quat(self):
        print('=== Testing parse_quat()')
        rot = quat.rotation()
        rot.parse_quat('') # TODO Add saved quaternion file
        
    def test_free_quat(self):
        print('=== Testing free_quat()')
        rot = quat.rotation()
        rot.quat_gen(4)
        rot.free_quat()
        rot.free_quat()
        self.assertIsNone(rot.num_rot)

class TestParams(unittest.TestCase):
    def configparams_test(self, param):
        # TODO Modify config file to make values non-default
        self.assertEqual(param.rank, 0)
        self.assertEqual(param.num_proc, 1)
        self.assertEqual(os.path.abspath(param.output_folder), os.path.abspath(recon_folder+'/data/'))
        self.assertEqual(os.path.abspath(param.log_fname), os.path.abspath(recon_folder+'/EMC.log'))
        self.assertEqual(param.need_scaling, 0)
        self.assertEqual(param.known_scale, 0)
        self.assertEqual(param.alpha, 0.)
        self.assertEqual(param.beta, 1.)
        self.assertEqual(param.beta_period, 100)
        self.assertEqual(param.beta_jump, 1.)
        self.assertEqual(param.sigmasq, 0.)
        
        # TODO Test with continue flag
        self.assertEqual(param.start_iter, 1)
        #self.assertEqual(param.current_iter, 0)
        #self.assertEqual(param.iteration, 0)
        #self.assertEqual(param.num_iter, 0)

    def test_generate_params(self):
        print('=== Testing generate_params()')
        param = params.params()
        param.generate_params(recon_folder+'/config.ini')
        self.configparams_test(param)
        param.generate_params(recon_folder+'/config.ini')

    def test_generate_output_dirs(self):
        print('=== Testing generate_output_dirs()')
        param = params.params()
        param.generate_params(recon_folder+'/config.ini')
        flist = [recon_folder+'/data/'+d for d in ['output', 'weights', 'orientations', 'scale', 'likelihood', 'mutualInfo']]
        [shutil.rmtree(d) for d in flist if os.path.exists(d)]
        param.generate_output_dirs()
        self.assertTrue(np.array([os.path.exists(d) for d in flist]).all())
        param.generate_output_dirs()

    def test_free_params(self):
        print('=== Testing free_params()')
        param = params.params()
        param.generate_params(recon_folder+'/config.ini')
        param.free_params()
        param.free_params()

class TestInterp(unittest.TestCase):
    def test_make_rot_quat(self):
        print('=== Testing make_rot_quat()')
        npt.assert_array_almost_equal(interp.make_rot_quat(np.array([1,0,0,0], dtype='f8')), np.identity(3))
        npt.assert_array_almost_equal(interp.make_rot_quat(0.5*np.array([1,-1,-1,-1], dtype='f8')), [[0,0,1],[1,0,0],[0,1,0]])
        npt.assert_array_almost_equal(interp.make_rot_quat(0.5*np.array([1,-1,-1,+1], dtype='f8')), [[0,1,0],[0,0,-1],[-1,0,0]])
        npt.assert_array_almost_equal(interp.make_rot_quat(0.5*np.array([1,+1,+1,-1], dtype='f8')), [[0,0,-1],[1,0,0],[0,-1,0]])
        npt.assert_array_almost_equal(interp.make_rot_quat(np.array([0,0,1/np.sqrt(2.),-1/np.sqrt(2.)], dtype='f8')), [[-1,0,0],[0,0,-1],[0,-1,0]])
        npt.assert_array_almost_equal(interp.make_rot_quat(np.array([1/np.sqrt(2.),0,1/np.sqrt(2.),0], dtype='f8')), [[0,0,-1],[0,1,0],[1,0,0]])
        npt.assert_array_almost_equal(interp.make_rot_quat(np.array([-1/np.sqrt(2.),1/np.sqrt(2.),0,0], dtype='f8')), [[1,0,0],[0,0,-1],[0,1,0]])

    def test_symmetrize_friedel(self):
        print('=== Testing symmetrize_friedel()')
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
        print('=== Testing rotate_model()')
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
        
        intens = 1.e-9*np.fromfile(recon_folder+'/data/intensities.bin').reshape(3*(145,))
        quat = np.array([np.sqrt(0.86),0.1,0.2,0.3])
        rot = interp.make_rot_quat(quat)
        rotmodel = np.zeros_like(intens)
        interp.rotate_model(rot, intens, rotmodel)
        npt.assert_array_almost_equal(rotmodel[100:103,100:103,100:103], [[[0.09167898, 0.03647299, 0.00707748], [0.12777323, 0.06260195, 0.02031047], [0.16907458, 0.09657051, 0.04333556]], [[0.06231541, 0.02152542, 0.00546131], [0.09500319, 0.04364314, 0.0141322], [0.13197364, 0.07370418, 0.03259982]], [[0.04283407, 0.01201542, 0.00398634], [0.06886472, 0.02839342, 0.00625303], [0.10017888, 0.05289991, 0.02144472]]])

    def test_slice_gen(self):
        print('=== Testing slice_gen()')
        det = detector.detector()
        det.parse_detector(recon_folder+'/data/det_sim.dat')
        intens = 1.e-9*np.fromfile(recon_folder+'/data/intensities.bin').reshape(3*(145,))
        view = np.zeros(det.num_pix)
        
        quat = np.array([1.,0,0,0])
        interp.slice_gen(quat, view, intens, det)
        self.assertAlmostEqual(view.mean(), 223.922218946)
        npt.assert_array_almost_equal(view[:5], [0.03021473, 0.02554173, 0.01861631, 0.01085438, 0.00459315])
        interp.slice_gen(quat, view, intens, det, rescale=1.)
        self.assertAlmostEqual(view.mean(), 1.86882056344)
        npt.assert_array_almost_equal(view[:5], [-3.49942586, -3.66744152, -3.98371703, -4.5231864, -5.38318919])
        
        quat = np.array([np.sqrt(0.86),0.1,0.2,0.3])
        interp.slice_gen(quat, view, intens, det)
        self.assertAlmostEqual(view.mean(), 184.449773553)
        npt.assert_array_almost_equal(view[:5], [0.00039123, 0.00014522, 0.00057308, 0.00185642, 0.00371838])
        interp.slice_gen(quat, view, intens, det, rescale=1.)
        self.assertAlmostEqual(view.mean(), 0.567310517859)
        npt.assert_array_almost_equal(view[:5], [-7.84620536, -8.83729446, -7.46449246, -6.28910363, -5.59446611])

    def test_slice_merge(self):
        print('=== Testing slice_merge()')
        det = detector.detector()
        det.parse_detector(recon_folder+'/data/det_sim.dat')
        view = np.ascontiguousarray(det.pixels[:,3])
        quat = np.array([np.sqrt(0.86),0.1,0.2,0.3])
        model = np.zeros(3*(145,))
        weight = np.zeros_like(model)
        interp.slice_merge(quat, view, model, weight, det)
        npt.assert_array_almost_equal(model, weight)
        npt.assert_array_almost_equal(model[103:106,68:71,82:85], [[[0.05970267, 0.86777407, 0.08261854], [0.29557584, 0.6624112, 0.00868513], [0.69197243, 0.40168333, 0.]], [[0., 0.69936496, 0.34398347], [0.07919628, 1.25519294, 0.0490487], [0.38575382, 0.65815609, 0.]], [[0., 0.45319003, 0.65207203], [0.01374454, 0.68324196, 0.25491581], [0.12366799, 0.84800088, 0.05462553]]])
        
        view = np.ascontiguousarray(det.pixels[:,3])*np.arange(det.num_pix)
        model = np.zeros(3*(145,))
        weight2 = np.zeros_like(model)
        interp.slice_merge(quat, view, model, weight2, det)
        npt.assert_array_almost_equal(weight, weight2)
        npt.assert_array_almost_equal(model[103:106,68:71,82:85], [[[480.23952805, 7053.79230354, 672.87605846], [2377.76518912, 5341.74335349, 70.74035788], [5519.30333337, 3220.70729727, 0.]], [[0., 5738.38868753, 2835.8821622], [640.53422865, 10226.00092727, 403.93064498], [3106.35276845, 5325.18474032, 0.]], [[0., 3752.37021246, 5424.77431879], [111.97675292, 5624.55664759, 2102.41717762], [1007.59124681, 6925.69283809, 450.55910447]]])

if __name__ == '__main__':
    print('Testing using recon folder: %s'%recon_folder)
    unittest.main(verbosity=0)

