import sys
import argparse
import multiprocessing as mp
import ctypes

import numpy as np
import h5py
from scipy import interpolate

class CCCalculator():
    def __init__(self, intens, n_angbins=40, mask_radius=20, interp_order=1):
        self.intens = intens
        self.size = intens.shape[-1]
        self.n_angbins = n_angbins
        self.mask_radius = mask_radius
        self.interp_order = interp_order
        self.cc_models = None

        self.get_samples(n_angbins, mask_radius)

    def get_samples(self, n_angbins, mask_radius):
        inds = np.arange(self.size) - self.size//2
        rads = inds.copy()[np.abs(inds) > mask_radius]
        x_matrix = np.zeros((n_angbins, self.size))

        angbin = np.pi / n_angbins
        self.x_vals = np.array([rads * np.cos(i*angbin) for i in range(n_angbins)])
        self.y_vals = np.array([rads * np.sin(i*angbin) for i in range(n_angbins)])

        self.intx = np.round(self.x_vals + self.size // 2).astype('i4')
        self.inty = np.round(self.y_vals + self.size // 2).astype('i4')

    def compare(self, intens1, intens2):
        if self.interp_order > 0:
            order = self.interp_order
            grid = np.arange(self.size) - self.size // 2
            fp1 = interpolate.RectBivariateSpline(grid, grid, intens1, kx=order, ky=order)
            fp2 = interpolate.RectBivariateSpline(grid, grid, intens2, kx=order, ky=order)

            z_vals1 = fp1(self.x_vals, self.y_vals, grid=False)
            z_vals2 = fp2(self.x_vals, self.y_vals, grid=False)
        elif self.interp_order == 0:
            z_vals1 = intens1[self.intx, self.inty]
            z_vals2 = intens2[self.intx, self.inty]

        cc = np.corrcoef(z_vals1, z_vals2)
        return cc[:self.x_vals.shape[0], self.x_vals.shape[0]:]

    def _mp_worker(self, intens, indices, cc_shared):
        n_models = intens.shape[0]
        irange = indices[:, 0]
        jrange = indices[:, 1]
        num = 0

        for i, j in zip(irange, jrange):
            ccs = self.compare(intens[i], intens[j])
            cc_shared[j*n_models + i] = ccs.max()
            num += 1
            if irange[0] == 0 and jrange[0] == 1:
                sys.stderr.write('\rC[%d,%d] = %f (%d/%d)   ' % (i, j, ccs.max(), num, len(irange)))
        if irange[0] == 0 and jrange[0] == 1:
            sys.stderr.write('\n')

    def run(self, fname_output=None, nproc=16):
        n_models = self.intens.shape[0]
        norm_intens = self.intens / self.intens.sum((1,2), keepdims=True)

        cc_shared = mp.Array(ctypes.c_double, n_models**2)

        ind = []
        # Get pairs to be processed
        for i in range(n_models):
            for j in range(i+1, n_models):
                ind.append([i,j])
        ind = np.array(ind)

        jobs = [mp.Process(target=self._mp_worker, args=(norm_intens, ind[i::nproc], cc_shared)) for i in range(nproc)]
        _ = [j.start() for j in jobs]
        _ = [j.join() for j in jobs]

        self.cc_models = np.frombuffer(cc_shared.get_obj()).reshape(n_models, n_models)
        self.cc_models[np.where(self.cc_models == 0)] = 1
        self.cc_models = np.minimum(self.cc_models, self.cc_models.T)

        if fname_output is not None:
            self.save_cc(fname_output)
        return self.cc_models

    def save_cc(self, fname_output):
        if self.cc_models is None:
            print('Calculate CC first using run()')
            return
        with h5py.File(fname_output, 'a') as h5f:
            if 'CL_CC' in h5f:
                del h5f['CL_CC']
            h5f['CL_CC/cc_matrix'] = self.cc_models
            h5f['CL_CC/n_angbins'] = self.n_angbins
            h5f['CL_CC/mask_radius'] = self.mask_radius
            h5f['CL_CC/interp_order'] = self.interp_order

def main():
    parser = argparse.ArgumentParser(description='Calculate Common-line CC matrix')
    parser.add_argument('intens_fname', 
                        help='Path to Dragonfly output file containing 2D intensity stack')
    parser.add_argument('-n', '--n_angbins', type=int, default=40,
                        help='Number of angular samples')
    parser.add_argument('-i', '--interp_order', type=int, default=1,
                        help='Interpolation order for radial line-outs')
    parser.add_argument('-m', '--mask_radius', type=int, default=20,
                        help='Radius of inner mask to be ignored')
    parser.add_argument('-o', '--output_fname', default=None,
                        help='Path to output file name (default: same as intens_fname)')
    parser.add_argument('-p', '--processes', type=int, default=16,
                        help='Number of multiprocessing processes')
    args = parser.parse_args()
    
    with h5py.File(args.intens_fname, 'r') as h5f:
        intens = h5f['intens'][:]
    calc = CCCalculator(intens,
                        n_angbins=args.n_angbins,
                        mask_radius=args.mask_radius,
                        interp_order=args.interp_order)
    calc.run(nproc=args.processes)

    if args.output_fname is None:
        args.output_fname = args.intens_fname
    calc.save_cc(args.output_fname)

if __name__ == '__main__':
    main()
