'''Matplotlib renderers used by autoplot frontends.'''

import matplotlib
import numpy as np

from .autoplot_core import align_models, get_default_normvecs, get_mode_grid_shape, get_normslice, normalize_highq, subtract_radmin

class VolumePlotter(object):
    def __init__(self, fig, output_parser):
        self.fig = fig
        self.canvas = fig.canvas
        self.parser = output_parser
        self.recon_type = output_parser.recon_type
        self.num_modes = output_parser.num_modes
        self.num_nonrot = output_parser.num_nonrot

        self.vol = None
        self.rots = None
        self.modes = None
        self.old_modenum = None
        self.main_subp = None
        self.imshow_args = None
        self.intrad = None
        self._init_xval = 0
        self._init_yval = 0
        self.normvecs = get_default_normvecs()

        if self.num_nonrot > 0 and self.parser.num_rot is None:
            raise ValueError('Need num_rot if nonrot modes are present')
        self.num_rot = self.parser.num_rot
        self.need_replot = False
        self.image_exists = False

    def parse(self, fname, modenum=0, rots=True):
        '''Parse volume defined in options panel.'''
        parsed = self.parser.parse_volume(fname, modenum=modenum, rots=rots)
        if parsed is None:
            return 0, 0, 0

        self.vol = parsed['vol']
        self.rots = parsed['rots']
        self.modes = parsed['modes']
        self.old_fname = parsed['fname']
        if self.num_modes > 1:
            self.old_modenum = modenum

        if self.recon_type == '3d':
            return parsed['fname'], parsed['size'], parsed['center']
        return parsed['fname'], self.num_modes, parsed['center']

    def plot(self, num, vrange, exponent, cmap):
        '''Plot volume on to self.fig.'''
        if self.vol is None:
            return
        rangemin, rangemax = tuple(vrange)
        self.imshow_args = {
            'cmap': cmap,
            'interpolation': 'none',
        }
        if exponent == 'log':
            self.imshow_args['norm'] = matplotlib.colors.SymLogNorm(linthresh=rangemax*1.e-2, vmin=rangemin, vmax=rangemax)
        else:
            self.imshow_args['norm'] = matplotlib.colors.PowerNorm(float(exponent), vmin=rangemin, vmax=rangemax)

        self.fig.clf()
        self.subplot_list = []
        if self.recon_type == '3d':
            for i in range(3):
                subp = self.fig.add_subplot(1, 3, i+1)
                vslice = get_normslice(self.vol, self.normvecs[i], num)
                subp.imshow(vslice, **self.imshow_args)
                subp.set_title(str(np.round(self.normvecs[i], 3)), y=1.01)
                subp.axis('off')
                self.subplot_list.append(subp)
        elif self.recon_type == '2d':
            tot_num_modes = self.num_modes + self.num_nonrot
            numx, numy = get_mode_grid_shape(self.num_modes, self.num_nonrot)
            total_numx = numx + int(np.ceil(numx / 2)) + 1

            gspec = matplotlib.gridspec.GridSpec(numy, total_numx)
            gspec.update(wspace=0.02, hspace=0.02)
            for mode in range(tot_num_modes):
                subp = self.fig.add_subplot(gspec[mode//numx, mode%numx])
                subp.imshow(self.vol[mode], **self.imshow_args)
                subp.text(0.05, 0.85, '%d' % mode, transform=subp.transAxes, fontsize=10, color='w')
                subp.axis('off')
                self.subplot_list.append(subp)
            self.main_subp = self.fig.add_subplot(gspec[:, numx:])
            self.main_subp.imshow(self.vol[num], **self.imshow_args)
            self.main_subp.set_title('Class %d' % num)
            self.main_subp.axis('off')

        self.canvas.draw()
        self.image_exists = True
        self.need_replot = False

    def update_mode(self, mode, vrange, exponent, cmap):
        if self.main_subp is None:
            return
        self.main_subp.clear()
        self.main_subp.imshow(self.vol[mode], **self.imshow_args)
        if self.rots is None:
            self.main_subp.set_title('Class %d' % mode)
        else:
            self.main_subp.set_title('Class %d (%d frames)' % (mode, (self.modes == mode).sum()))
        self.main_subp.axis('off')
        self.canvas.draw()

    def subtract_radmin(self):
        if self.vol is None:
            return
        self.vol = subtract_radmin(self.vol, self.recon_type, self.num_modes)

    def normalize_highq(self):
        if self.vol is None:
            return
        self.vol = normalize_highq(self.vol, self.recon_type)

    def align_models(self):
        if self.vol is None:
            return
        self.vol = align_models(self.vol, self.recon_type)

class LogPlotter(object):
    def __init__(self, fig, output_parser):
        self.fig = fig
        self.canvas = fig.canvas
        self.parser = output_parser
        self.rots = None

    def plot(self, fname, cmap):
        '''Plot various metrics from the log file as a function of iteration.'''
        all_lines, loglines = self.parser.read_log(fname)
        if loglines is None:
            return all_lines

        iternum = loglines[:, 0].astype('i4')
        num_rot = loglines[:, 5].astype('i4')
        beta = loglines[:, 6].astype('f8')
        self.num_rot_change = np.append(np.where(np.diff(num_rot) != 0)[0], num_rot.shape[0])
        self.beta_change = np.where(np.diff(beta) != 0.)[0]

        o_array = self.parser.get_orientations(loglines, self.num_rot_change)

        self.fig.clf()
        grid = matplotlib.gridspec.GridSpec(2, 3, wspace=0.3, hspace=0.2)
        grid.update(left=0.05, right=0.99, hspace=0.2, wspace=0.3)

        self._add_logplot(grid[:, 0], iternum, loglines[:, 2], 'RMS change')
        self._add_logplot(grid[0, 1], iternum, loglines[:, 3], r'Mutual info. $I(K,\Omega | W)$', yscale='linear')
        self._add_logplot(grid[1, 1], iternum[1:], loglines[1:, 4], 'Avg log-likelihood', yscale='symlog')

        if o_array is not None and len(loglines) > 1:
            subp = self.fig.add_subplot(grid[:, 2])
            o_array = o_array[o_array[:, -1] >= 0]
            shp = o_array.shape
            subp.imshow(o_array, aspect=(1.*shp[1]/shp[0]), extent=[1, shp[1], shp[0], 0], cmap=cmap)
            subp.get_yaxis().set_ticks([])
            subp.set_xlabel('Iteration')
            subp.set_ylabel('Pattern number (sorted)')
            subp.set_title('Most likely orientations of data\n(sorted/colored by last iteration)')

        grid.tight_layout(self.fig)
        self.canvas.draw()
        return all_lines

    def _add_logplot(self, gridpos, xval, yval, title='', yscale='log'):
        subp = self.fig.add_subplot(gridpos)
        subp.plot(xval, yval.astype('f8'), 'o-')
        subp.set_yscale(yscale)
        subp.set_xlabel('Iteration')
        subp.set_ylabel(title)
        ylim = subp.get_ylim()
        subp.set_ylim(ylim)
        subp.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        for i in self.beta_change:
            subp.plot([i+1-0.1, i+1-0.1], ylim, '--', color='w', lw=1)
        for i in self.num_rot_change[:-1]:
            subp.plot([i+1+0.1, i+1+0.1], ylim, '--', color='tab:orange', lw=1)
