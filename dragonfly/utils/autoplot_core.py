'''Parser and state helpers used by autoplot frontends.'''

import os
import sys
from types import SimpleNamespace

import dragonfly
import h5py
import numpy as np
from scipy import ndimage

from .py_src import py_utils, read_config, slices


def get_default_normvecs():
    return np.identity(3)

def get_mode_grid_shape(num_modes, num_nonrot=0):
    tot_num_modes = num_modes + num_nonrot
    numx = int(np.ceil(2.*np.sqrt(tot_num_modes / 2.)))
    numy = int(np.ceil(tot_num_modes / float(numx)))
    return numx, numy

def get_normslice(vol, vec, layernum):
    vec = np.asarray(vec, dtype='f8')
    vec = vec / np.linalg.norm(vec)

    size = vol.shape[-1]
    ind = np.arange(size) - size // 2
    i, j = np.meshgrid(ind, ind, indexing='ij')

    if vec[0] < 0.95:
        v2 = np.cross(vec, [1., 0., 0.])
    else:
        v2 = np.cross(vec, [0., 1., 0.])
    v2 /= np.linalg.norm(v2)
    v1 = np.cross(vec, v2)

    coords = np.outer(v1, i) + np.outer(v2, j) + size // 2
    coords = (coords.T + vec * (layernum-size//2)).T
    coords = coords.reshape(3, size, size)
    return ndimage.map_coordinates(vol, coords, order=1, prefilter=False)

def _get_intrad(vol, recon_type):
    size = vol.shape[1]
    cen = size // 2
    ind = np.arange(size, dtype='f4') - cen
    if recon_type == '2d':
        x, y = np.meshgrid(ind, ind, indexing='ij')
        intrad = np.sqrt(x**2 + y**2).astype('i4')
        return intrad, x, y

    x, y, z = np.meshgrid(ind, ind, ind, indexing='ij')
    intrad = np.sqrt(x**2 + y**2 + z**2).astype('i4')
    return intrad, x, y, z

def subtract_radmin(vol, recon_type, num_modes):
    intrad = _get_intrad(vol, recon_type)[0]
    if recon_type == '2d':
        out = vol.copy()
        for mode in range(num_modes):
            radmin = np.ones(intrad.max()+1) * 1e20
            np.minimum.at(radmin, intrad, out[mode])
            out[mode] -= radmin[intrad]
        return out

    out = vol.copy()
    radmin = np.ones(intrad.max()+1) * 1e20
    np.minimum.at(radmin, intrad, out)
    out -= radmin[intrad]
    return out

def normalize_highq(vol, recon_type):
    if recon_type != '2d':
        raise ValueError('High q normalization only implemented for 2D EMC')
    intrad = _get_intrad(vol, recon_type)[0]
    out = vol.copy()
    hsize = out.shape[-1] // 2
    radsel = (intrad < hsize - 3) & (intrad > 0.9 * hsize)
    highq_vals = out[:, radsel].mean(1)
    out /= highq_vals[:, None, None]
    return out

def align_models(vol, recon_type):
    if recon_type != '2d':
        raise ValueError('Alignment currently implemented only for 2D EMC')
    intrad, x, y = _get_intrad(vol, recon_type)
    _ = intrad
    imat = np.array([[x**2, x*y], [x*y, y**2]])
    angles = [np.arctan2(*np.linalg.eigh((imat*vol[i]).sum((2, 3))).eigenvectors[0])*180/np.pi for i in range(len(vol))]
    return np.array([ndimage.rotate(vol[i], 90+angles[i], order=1, reshape=False) for i in range(len(vol))])

class OutputParser(object):
    '''Shared parser for EMC output volumes, logs, and config metadata.'''
    def __init__(self, recon_type='3d', num_modes=1, num_nonrot=0, num_rot=None, folder='data/'):
        self.recon_type = recon_type
        self.num_modes = num_modes
        self.num_nonrot = num_nonrot
        self.num_rot = num_rot
        self.folder = folder

    @classmethod
    def from_config(cls, config_fname):
        config = read_config.MyConfigParser()
        config.read(config_fname)
        return cls(
            recon_type=config.get('emc', 'recon_type', fallback='3d').lower(),
            num_modes=config.getint('emc', 'num_modes', fallback=1),
            num_nonrot=config.getint('emc', 'num_nonrot', fallback=0),
            num_rot=config.getint('emc', 'num_rot', fallback=-1),
            folder=config.get_filename('emc', 'output_folder', fallback='data/')
        ), config.get_filename('emc', 'log_file', fallback='logs/EMC.log')

    def gen_model_fname(self, num):
        h5_fname = self.folder+'/output_%.3d.h5' % num
        if os.path.isfile(h5_fname):
            return h5_fname
        return self.folder+'/output/intens_%.3d.bin' % num

    def get_latest_iteration(self, fname):
        if not os.path.isfile(fname):
            return 0
        with open(fname, 'r') as fptr:
            try:
                last_line = fptr.readlines()[-1].rstrip().split()
            except IndexError:
                return 0
        try:
            return int(last_line[0])
        except (IndexError, ValueError):
            return 0

    def parse_volume(self, fname, modenum=0, rots=True):
        '''Parse an EMC output file and return volume metadata for plotting.'''
        if self.num_nonrot > 0 and self.num_rot is None:
            raise ValueError('Need num_rot if nonrot modes are present')

        rots_array = None
        modes = None
        if not os.path.isfile(fname):
            sys.stderr.write("Unable to open %s\n" % fname)
            return None

        if h5py.is_hdf5(fname):
            h5_output = True
            with h5py.File(fname, 'r') as fptr:
                if self.recon_type == '3d':
                    vol = fptr['intens'][modenum]
                else:
                    vol = fptr['intens'][:]
                    if self.num_modes == 1:
                        vol = vol[0]
                if rots:
                    try:
                        rots_array = fptr['orientations'][:]
                    except KeyError:
                        print('No orientations dataset in', fname)
            size = vol.shape[-1]
        else:
            h5_output = False
            vol = np.fromfile(fname, dtype='f8')
            if rots:
                try:
                    # Assuming fname is <out_folder>/output/output_???.bin
                    iternum = int(fname[-7:-4])
                    out_folder = fname[:-21]
                    rots_array = np.fromfile(out_folder+'/orientations/orientations_%.3d.bin' % iternum, '=i4')
                except (ValueError, IOError):
                    rots_array = None

        if self.recon_type == '3d':
            if not h5_output:
                size = int(np.ceil(np.power(len(vol)/self.num_modes, 1./3.)))
                if self.num_modes > 1:
                    vol = vol[modenum*size**3:(modenum+1)*size**3].reshape(size, size, size)
                else:
                    vol = vol.reshape(size, size, size)
            center = size // 2
        else:
            if not h5_output:
                size = int(np.ceil(np.power(len(vol)/self.num_modes, 1./2.)))
                vol = vol.reshape(self.num_modes, size, size)
            center = 0

        if self.num_modes > 1 and rots_array is not None:
            rotind = rots_array // self.num_modes
            modes = rots_array % self.num_modes
            modes[rots_array < 0] = -1
            if self.num_nonrot > 0:
                modes[rotind >= self.num_rot] = rots_array[rotind >= self.num_rot] - self.num_modes * (self.num_rot - 1)

        return {
            'fname': fname,
            'vol': vol,
            'rots': rots_array,
            'modes': modes,
            'size': size,
            'center': center,
        }

    def read_log(self, fname):
        '''Read the EMC log file and return raw text with parsed iteration rows.'''
        with open(fname, 'r') as fptr:
            all_lines = fptr.readlines()
        lines = [line.rstrip().split() for line in all_lines]
        loglines = [line for line in lines if len(line) > 0 and line[0].isdigit()]
        if len(loglines) == 0:
            return ''.join(all_lines), None
        return ''.join(all_lines), np.array(loglines)

    def get_orientations(self, loglines, num_rot_change):
        orient = []
        for i in range(len(loglines)):
            h5_fname = self.folder+'/output_%.3d.h5' % (i+1)
            bin_fname = self.folder+'/orientations/orientations_%.3d.bin' % (i+1)
            if not os.path.isfile(h5_fname) and not os.path.isfile(bin_fname):
                print('Missing intermediate iterations')
                return None

            if os.path.isfile(h5_fname):
                with h5py.File(h5_fname, 'r') as fptr:
                    orient.append(fptr['orientations'][:])
            else:
                with open(bin_fname, 'r') as fptr:
                    orient.append(np.fromfile(fptr, '=i4'))

        olengths = np.array([len(ori) for ori in orient])
        max_length = olengths.max()

        # Sort o_array by the last iteration which has the same number of orientations
        o_array = np.array([np.pad(o, ((max_length-len(o), 0)), 'constant', constant_values=-1)
                            for o in orient]).astype('f8')
        istart = 0
        for istop in num_rot_change:
            sorter = o_array[istop-1].argsort()
            for index in np.arange(istart, istop):
                o_array[index] = o_array[index][sorter]
            istart = istop
        return o_array.T

class AutoplotController(object):
    '''Backend-neutral state and orchestration for autoplot frontends.'''
    def __init__(self, config='config.ini', model=None):
        self.model_name = model
        self.mode_select = False
        self.selected_modes = set()
        self.num_good = 0
        self.max_iternum = 0
        self.old_fname = None
        self.old_modenum = 0
        self.output_parser = None
        self.logfname = None
        self.folder = None
        self.recon_type = None
        self.num_modes = None
        self.num_nonrot = None
        self.num_rot = None
        self.config = None
        self.load_config(config)

    def load_config(self, config_fname):
        self.output_parser, self.logfname = OutputParser.from_config(config_fname)
        self.folder = self.output_parser.folder
        self.recon_type = self.output_parser.recon_type
        self.num_modes = self.output_parser.num_modes
        self.num_nonrot = self.output_parser.num_nonrot
        self.num_rot = self.output_parser.num_rot
        self.config = config_fname
        self.max_iternum = 0
        self.old_fname = None
        self.old_modenum = 0
        self.set_mode_selection(False)

    def get_initial_volume_fname(self):
        if self.model_name is not None:
            return self.model_name
        return self.folder + '/output_001.h5'

    def gen_model_fname(self, num):
        return self.output_parser.gen_model_fname(num)

    def plan_plot(self, current_fname, current_modenum, force=False, image_exists=False,
                  need_replot=False):
        if force or not image_exists or self.old_fname != current_fname:
            if self.num_modes > 1:
                return 'parse', current_modenum
            return 'parse', 0
        if self.num_modes > 1 and current_modenum != self.old_modenum:
            return 'parse', current_modenum
        if need_replot:
            return 'replot', current_modenum
        return 'none', current_modenum

    def record_parse(self, fname, modenum):
        self.old_fname = fname
        self.old_modenum = modenum

    def record_mode(self, modenum):
        self.old_modenum = modenum

    def check_for_new_iteration(self, logfname):
        iteration = self.output_parser.get_latest_iteration(logfname)
        if iteration > 0 and self.max_iternum != iteration:
            self.max_iternum = iteration
            return {
                'updated': True,
                'iteration': iteration,
                'fname': self.gen_model_fname(iteration),
            }
        return {
            'updated': False,
            'iteration': iteration,
            'fname': self.gen_model_fname(iteration) if iteration > 0 else None,
        }

    def set_mode_selection(self, status):
        self.mode_select = status
        if not status:
            self.selected_modes.clear()
            self.num_good = 0

    def is_mode_selected(self, mode):
        return mode in self.selected_modes

    def toggle_selected_mode(self, mode, modes):
        if modes is None:
            return False

        count = int((modes == mode).sum())
        if mode in self.selected_modes:
            self.selected_modes.remove(mode)
            self.num_good -= count
            return False

        self.selected_modes.add(mode)
        self.num_good += count
        return True

    def generate_blacklist(self, modes):
        if modes is None:
            return None
        blist = np.ones(modes.shape, dtype='u1')
        for mode in self.selected_modes:
            blist[modes == mode] = 0
        return blist

class FrameviewerController(object):
    '''Backend-neutral frame loading and filtering for frameviewer frontends.'''
    def __init__(self, config_file, mask=False):
        self.config_file = config_file
        self.mask = mask
        self._ctx = SimpleNamespace(config_file=config_file)
        read_config.read_gui_config(self._ctx, 'emc')
        py_utils.gen_det_and_emc(self._ctx, classifier=False, mask=mask)
        self.emc_reader = self._ctx.emc_reader
        self.geom = self._ctx.geom
        self.blacklist = self._ctx.blacklist
        self.log_fname = self._ctx.log_fname
        self.output_folder = self._ctx.output_folder
        self.output_parser, _ = OutputParser.from_config(config_file)
        self.slice_generator = slices.SliceGenerator(config_file)
        self.recon_type = self.output_parser.recon_type
        self.num_modes = self.output_parser.num_modes

    @property
    def num_frames(self):
        return self.emc_reader.num_frames

    def get_mode_frames(self, output_fname, mode):
        parsed = self.output_parser.parse_volume(output_fname, modenum=mode, rots=True)
        if parsed is None or parsed['modes'] is None:
            return None
        return np.where(parsed['modes'] == mode)[0]

    def get_available_frames(self, output_fname=None, mode=None, skip_bad=False):
        frames = np.arange(self.num_frames)
        if skip_bad and self.blacklist is not None:
            frames = frames[self.blacklist == 0]
        if output_fname is not None and mode is not None:
            mode_frames = self.get_mode_frames(output_fname, mode)
            if mode_frames is not None:
                frame_set = set(mode_frames.tolist())
                frames = np.array([frame for frame in frames if frame in frame_set], dtype='i4')
        return frames

    def get_frame_image(self, frame_num, sym=False):
        frame = self.emc_reader.get_frame(frame_num, zoomed=True, sym=sym, avg=True)
        det = self.emc_reader.flist[self.emc_reader._get_file_and_frame(frame_num)[0]]['det']
        cen = det.get_assembled_cen(zoomed=True, sym=sym)
        return frame, cen

    def get_frame_title(self, frame, frame_num):
        title = '%d photons' % frame.sum()
        if self.blacklist is not None and self.blacklist[frame_num] == 1:
            title += ' (bad frame)'
        return title

    def get_compare_slice(self, iteration, frame_num, sym=False):
        if iteration is None or iteration <= 0:
            return None, None
        return self.slice_generator.get_slice(iteration, frame_num, zoomed=True, sym=sym, avg=True)
