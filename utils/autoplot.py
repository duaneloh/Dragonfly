#!/usr/bin/env python

'''Module containing Dragonfly Progress Viewer'''

from __future__ import print_function
import sys
import os
import argparse
import numpy as np
from scipy import ndimage
import h5py
try:
    from PyQt5 import QtCore, QtWidgets, QtGui # pylint: disable=import-error
    import matplotlib
    matplotlib.use('qt5agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvas # pylint: disable=no-name-in-module
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    os.environ['QT_API'] = 'pyqt5'
except ImportError:
    import sip
    sip.setapi('QString', 2)
    from PyQt4 import QtCore, QtGui # pylint: disable=import-error
    from PyQt4 import QtGui as QtWidgets # pylint: disable=import-error
    import matplotlib
    matplotlib.use('qt4agg')
    from matplotlib.backends.backend_qt4agg import FigureCanvas # pylint: disable=no-name-in-module
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    os.environ['QT_API'] = 'pyqt'
from py_src import read_config
import frameviewer
from py_src import gui_utils
from py_src import clpca
from py_src import phaser_gui

class MySpinBox(QtWidgets.QSpinBox):
    '''Overriding QSpinBox to update need_replot'''
    def __init__(self, parent, *args, **kwargs):
        super(MySpinBox, self).__init__(parent, *args, **kwargs)
        self.parent = parent

    def stepBy(self, steps): # pylint: disable=C0103
        '''Override stepBy to do range checking and set need_replot=True'''
        target_value = self.value() + steps
        if target_value < self.minimum():
            self.setValue(self.minimum())
        elif target_value > self.maximum():
            self.setValue(self.maximum())
        else:
            self.setValue(target_value)
        self.parent.need_replot = True

class MyFrameviewer(frameviewer.Frameviewer):
    windowClosed = QtCore.pyqtSignal()

    def __init__(self, config_file, mode, numlist):
        super(MyFrameviewer, self).__init__(config_file, mask=True, noscroll=True)
        self.mode = mode
        self.numlist = numlist

        fp_layout = self.frame_panel.layout()
        fp_count = fp_layout.count()
        line = fp_layout.takeAt(fp_count-1)
        hbox = QtWidgets.QHBoxLayout()
        line.insertLayout(1, hbox)
        gui_utils.add_scroll_hbox(self, hbox)
        if self.mode >= 0:
            self.label = QtWidgets.QLabel('Class %d frames'%self.mode)
            hbox.addWidget(self.label)
        fp_layout.addLayout(line)

        self._next_frame()

    def _next_frame(self):
        if self.mode == -1:
            self.frame_panel._next_frame()
        else:
            curr = int(self.frame_panel.numstr.text())
            ind = np.searchsorted(self.numlist, curr, side='left')
            if curr == self.numlist[ind]:
                ind += 1
            if ind > len(self.numlist) - 1:
                ind = len(self.numlist) - 1
            num = self.numlist[ind]
            if num < self.frame_panel.emc_reader.num_frames:
                self.frame_panel.numstr.setText(str(num))
                self.frame_panel.plot_frame()

    def _prev_frame(self):
        if self.mode == -1:
            self.frame_panel._prev_frame()
        else:
            curr = int(self.frame_panel.numstr.text())
            ind = np.searchsorted(self.numlist, curr, side='left') - 1
            if ind < 0:
                ind = 0
            num = self.numlist[ind]
            if num > -1:
                self.frame_panel.numstr.setText(str(num))
                self.frame_panel.plot_frame()

    def _rand_frame(self):
        if self.mode == -1:
            self.frame_panel._rand_frame()
        else:
            curr = int(self.frame_panel.numstr.text())
            ind = np.searchsorted(self.numlist, curr, side='left')
            if curr == self.numlist[ind]:
                ind += 1
            if ind > len(self.numlist) - 1:
                ind = len(self.numlist) - 1
            num = self.numlist[np.random.randint(len(self.numlist))]
            self.frame_panel.numstr.setText(str(num))
            self.frame_panel.plot_frame()

    def closeEvent(self, event):
        self.windowClosed.emit()
        event.accept()

class NormVecUpdater(QtWidgets.QDialog):
    def __init__(self, old_normvec, parent=None):
        super(NormVecUpdater, self).__init__(parent)
        self.vec = old_normvec
        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel('Update normal vector (need not be unit vector):', self)
        layout.addWidget(label)

        line = QtWidgets.QHBoxLayout()
        layout.addLayout(line)
        self.v0 = QtWidgets.QLineEdit('%f'%old_normvec[0], self)
        line.addWidget(self.v0)
        self.v1 = QtWidgets.QLineEdit('%f'%old_normvec[1], self)
        line.addWidget(self.v1)
        self.v2 = QtWidgets.QLineEdit('%f'%old_normvec[2], self)
        line.addWidget(self.v2)

        line = QtWidgets.QHBoxLayout()
        layout.addLayout(line)
        button = QtWidgets.QPushButton('Update', self)
        button.clicked.connect(self._update)
        line.addWidget(button)
        button = QtWidgets.QPushButton('Cancel', self)
        button.clicked.connect(self.close)
        line.addWidget(button)
        self.exec_()

    def _update(self):
        self.vec = np.array([float(self.v0.text()), float(self.v1.text()), float(self.v2.text())])
        self.close()

class VolumePlotter(object):
    def __init__(self, fig, recon_type='3d', num_modes=1, num_nonrot=0, num_rot=None):
        self.fig = fig
        self.canvas = fig.canvas
        self.recon_type = recon_type
        self.num_modes = num_modes
        self.num_nonrot = num_nonrot

        self.vol = None
        self.rots = None
        self.modes = None
        self.old_modenum = None
        self.main_subp = None
        self.imshow_args = None
        self.intrad = None
        self._init_xval = 0
        self._init_yval = 0
        self.normvecs = np.identity(3)

        if self.num_nonrot > 0 and num_rot is None:
            raise ValueError('Need num_rot if nonrot modes are present')
        self.num_rot = num_rot
        self.need_replot = False
        self.image_exists = False

    def parse(self, fname, modenum=0, rots=True):
        '''Parse volume defined in options panel
        Can be either 3D volume or 2d slice stack depending on mode in config file
        '''
        if os.path.isfile(fname):
            if h5py.is_hdf5(fname):
                h5_output = True
                with h5py.File(fname, 'r') as f:
                    if self.recon_type == '3d':
                        self.vol = f['intens'][modenum]
                    else:
                        self.vol = f['intens'][:]
                        if self.num_modes == 1:
                            self.vol = self.vol[0]
                    if rots:
                        try:
                            self.rots = f['orientations'][:]
                        except KeyError:
                            print('No orientations dataset in', fname)
                            self.rots = None
                size = self.vol.shape[-1]
            else:
                h5_output = False
                self.vol = np.fromfile(fname, dtype='f8')
                if rots:
                    try:
                        # Assuming fname is <out_folder>/output/output_???.bin
                        iternum = int(fname[-7:-4])
                        out_folder = fname[:-21]
                        self.rots = np.fromfile(out_folder+'/orientations/orientations_%.3d.bin'%iternum, '=i4')
                    except (ValueError, IOError):
                        #print('No orientations found for iteration %d' % iternum)
                        self.rots = None
        else:
            sys.stderr.write("Unable to open %s\n"%fname)
            return 0, 0, 0

        if self.recon_type == '3d':
            if not h5_output:
                size = int(np.ceil(np.power(len(self.vol)/self.num_modes, 1./3.)))
                if self.num_modes > 1:
                    self.vol = self.vol[modenum*size**3:(modenum+1)*size**3].reshape(size, size, size)
                else:
                    self.vol = self.vol.reshape(size, size, size)
            center = size // 2
            return_val = (fname, size, center)
        else:
            if not h5_output:
                size = int(np.ceil(np.power(len(self.vol)/self.num_modes, 1./2.)))
                self.vol = self.vol.reshape(self.num_modes, size, size)
            center = 0
            return_val = (fname, self.num_modes, center)

        self.old_fname = fname
        if self.num_modes > 1:
            self.old_modenum = modenum
            if self.rots is not None:
                rotind = self.rots // self.num_modes
                self.modes = self.rots % self.num_modes
                self.modes[self.rots < 0] = -1
                if self.num_nonrot > 0:
                    self.modes[rotind >= self.num_rot] = self.rots[rotind >= self.num_rot] - self.num_modes * (self.num_rot - 1)
        return return_val

    def plot(self, num, vrange, exponent, cmap):
        '''Plot volume on to self.fig
        In normal 3D mode, this means 3 orthogonal slices passing through a given layer number
        In 2D mode, this means showing all classes and adding the ability to click to zoom
        one class at a time.

        The color map and color scale range can be set in the options panel
        '''
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
                vslice = self._get_normslice(self.normvecs[i], num)
                subp.imshow(vslice, **self.imshow_args)
                subp.set_title(str(np.round(self.normvecs[i], 3)), y=1.01)
                subp.axis('off')
                self.subplot_list.append(subp)
        elif self.recon_type == '2d':
            tot_num_modes = self.num_modes + self.num_nonrot
            numx = int(np.ceil(2.*np.sqrt(tot_num_modes / 2.)))
            numy = int(np.ceil(tot_num_modes / float(numx)))
            total_numx = numx + int(np.ceil(numx / 2)) + 1

            gspec = matplotlib.gridspec.GridSpec(numy, total_numx)
            gspec.update(wspace=0.02, hspace=0.02)
            for mode in range(tot_num_modes):
                subp = self.fig.add_subplot(gspec[mode//numx, mode%numx])
                subp.imshow(self.vol[mode], **self.imshow_args)
                subp.text(0.05, 0.85, '%d'%mode, transform=subp.transAxes, fontsize=10, color='w')
                subp.axis('off')
                self.subplot_list.append(subp)
            self.main_subp = self.fig.add_subplot(gspec[:, numx:])
            self.main_subp.imshow(self.vol[num], **self.imshow_args)
            self.main_subp.set_title('Class %d'%num)
            self.main_subp.axis('off')

        self.canvas.draw()
        self.image_exists = True
        self.need_replot = False

    def update_mode(self, mode, vrange, exponent, cmap):
        if self.main_subp is None:
            return
        rangemin, rangemax = tuple(vrange)
        self.main_subp.clear()
        self.main_subp.imshow(self.vol[mode], **self.imshow_args)
        if self.rots is None:
            self.main_subp.set_title('Class %d'%mode)
        else:
            self.main_subp.set_title('Class %d (%d frames)'%(mode, (self.modes == mode).sum()))
        self.main_subp.axis('off')
        self.canvas.draw()

    def _get_intrad(self):
        if self.intrad is not None and self.size == self.vol.shape[1]:
            return

        self.size = self.vol.shape[1]
        cen = self.size // 2
        if self.recon_type == '2d':
            self.x, self.y = np.indices(self.vol[0].shape)
            self.x -= cen
            self.y -= cen
            self.intrad = np.sqrt(self.x**2 + self.y**2).astype('i4')
        elif self.recon_type == '3d':
            self.x, self.y, self.z = np.indices(self.vol.shape)
            self.x -= cen
            self.y -= cen
            self.z -= cen
            self.intrad = np.sqrt(self.x**2 + self.y**2 + self.z**2).astype('i4')

    def _get_normslice(self, vec, layernum):
        vec /= np.linalg.norm(vec)

        size = self.vol.shape[-1]
        ind = np.arange(size) - size // 2
        i, j = np.meshgrid(ind, ind, indexing='ij')

        # If vector is not parallel to (1,0,0) use v2 as [vec x (1,0,0)]
        # Else use v2 as [vec x (0,1,0)]
        # v1 will then be [vec x v2]
        if vec[0] < 0.95:
            v2 = np.cross(vec, [1.,0.,0.])
        else:
            v2 = np.cross(vec, [0.,1.,0.])
        v2 /= np.linalg.norm(v2)
        v1 = np.cross(vec, v2)

        coords = np.outer(v1, i) + np.outer(v2, j) + size // 2
        coords = (coords.T + vec * (layernum-size//2)).T
        coords = coords.reshape(3, size, size)
        return ndimage.map_coordinates(self.vol, coords, order=1, prefilter=False)

    def subtract_radmin(self):
        if self.vol is None:
            return
        self._get_intrad()
        if self.recon_type == '2d':
            for m in range(self.num_modes):
                radmin = np.ones(self.intrad.max()+1) * 1e20
                np.minimum.at(radmin, self.intrad, self.vol[m])
                self.vol[m] -= radmin[self.intrad]
        else:
            radmin = np.ones(self.intrad.max()+1) * 1e20
            np.minimum.at(radmin, self.intrad, self.vol)
            self.vol -= radmin[self.intrad]

class LogPlotter(object):
    def __init__(self, fig, folder='data/'):
        self.fig = fig
        self.canvas = fig.canvas
        self.folder = folder
        self.rots = None

    def plot(self, fname, cmap):
        '''Plot various metrics from the log file as a function of iteration
        Metrics are:
            RMS change of 3D volume
            Average mutual information between frames and orientations
            Average log likelihood
            Variation of most likeliy orientation index for each frame
        '''
        # Read log file to get log lines (one for each completed iteration)
        with open(fname, 'r') as fptr:
            all_lines = fptr.readlines()

            lines = [l.rstrip().split() for l in all_lines]
            loglines = [line for line in lines if len(line) > 0 and line[0].isdigit()]

        loglines = np.array(loglines)
        if len(loglines) == 0:
            return

        iternum = loglines[:, 0].astype('i4')
        num_rot = loglines[:, 5].astype('i4')
        beta = loglines[:, 6].astype('f8')
        self.num_rot_change = np.append(np.where(np.diff(num_rot) != 0)[0], num_rot.shape[0])
        self.beta_change = np.where(np.diff(beta) != 0.)[0]

        o_array = self._get_orient(loglines)

        self.fig.clf()
        grid = matplotlib.gridspec.GridSpec(2, 3, wspace=0.3, hspace=0.2)
        grid.update(left=0.05, right=0.99, hspace=0.2, wspace=0.3)

        self._add_logplot(grid[:, 0], iternum, loglines[:, 2],
                          'RMS change')
        self._add_logplot(grid[0, 1], iternum, loglines[:, 3],
                          r'Mutual info. $I(K,\Omega | W)$', yscale='linear')

        self._add_logplot(grid[1, 1], iternum[1:], loglines[1:, 4],
                          'Avg log-likelihood', yscale='symlog')

        # Plot most likely orientation convergence plot
        if o_array is not None and len(loglines) > 1:
            subp = self.fig.add_subplot(grid[:, 2])
            o_array = o_array[o_array[:, -1] >= 0]
            shp = o_array.shape
            subp.imshow(o_array, aspect=(1.*shp[1]/shp[0]), extent=[1, shp[1], shp[0], 0],
                        cmap=cmap)
            subp.get_yaxis().set_ticks([])
            subp.set_xlabel('Iteration')
            subp.set_ylabel('Pattern number (sorted)')
            subp.set_title('Most likely orientations of data\n(sorted/colored by last iteration)')

        grid.tight_layout(self.fig)
        self.canvas.draw()
        return ''.join(all_lines)

    def _get_orient(self, loglines):
        orient = []

        # Read orientation files for the first n iterations
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
        for i, istop in enumerate(self.num_rot_change):
            sorter = o_array[istop-1].argsort()
            for index in np.arange(istart, istop):
                o_array[index] = o_array[index][sorter]
            istart = istop
        o_array = o_array.T

        return o_array

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

class ProgressViewer(QtWidgets.QMainWindow):
    '''GUI to track progress of EMC reconstruction
    Shows orthogonal volumes slices, plots of metrics vs iteration and log file
    Can periodically poll log file for updates and automatically update plots

    Can also be used to view slices through other 3D volumes using the '-f' option
    '''
    def __init__(self, config='config.ini', model=None):
        super(ProgressViewer, self).__init__()
        self.config = config
        self.model_name = model
        self.mode_select = False
        self.num_good = 0
        plt.style.use('dark_background')
        self.settings = QtCore.QSettings('DragonflyProgressViewer')

        self.beta_change = self.num_rot_change = []
        self.checker = QtCore.QTimer(self)

        self._read_config(config)
        self._init_ui()
        if model is not None:
            self._parse_and_plot(rots=False)
        self.old_fname = self.fname.text()
        self.fviewer = None
        self.clpca = None
        self.phaser2d = None

    def _init_ui(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'py_src/style.css'), 'r') as f:
            self.css = f.read()
            self.setStyleSheet(self.css)
        self.setWindowTitle('Dragonfly Progress Viewer')
        self.setGeometry(100, 100, 1600, 800)
        overall = QtWidgets.QWidget()
        self.setCentralWidget(overall)
        layout = QtWidgets.QHBoxLayout(overall)
        layout.setContentsMargins(0, 0, 0, 0)

        self._init_menubar()

        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.main_splitter.setObjectName('frame')
        layout.addWidget(self.main_splitter)
        self._init_main()

        self.show()

    def _init_main(self):
        for i in range(self.main_splitter.count()):
            widget = self.main_splitter.widget(i)
            widget.hide()
            del widget

        self.plot_splitter = self._init_plotarea()
        self.options_widget = self._init_optionsarea()

        self.main_splitter.addWidget(self.plot_splitter)
        self.main_splitter.addWidget(self.options_widget)

        self.max_iternum = 0
        self._check_for_new()

    def _init_menubar(self):
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)

        # File Menu
        filemenu = menubar.addMenu('&File')
        action = filemenu.addAction('Load &Volume')
        action.triggered.connect(self._load_volume)
        action.setToolTip('Load 3D volume (h5 or bin)')
        action = filemenu.addAction('Load &Config')
        action.triggered.connect(self._refresh_gui)
        action.setToolTip('Refresh GUI with new config file')
        action = filemenu.addAction('&Quit')
        action.triggered.connect(self.close)

        # Image Menu
        imagemenu = menubar.addMenu('&Image')
        action = imagemenu.addAction('&Save Slices Image')
        action.triggered.connect(self._save_plot)
        action.setToolTip('Save current plot of slices as image')
        action = imagemenu.addAction('Save Log &Plot')
        action.triggered.connect(self._save_log_plot)
        action.setToolTip('Save panel of metrics plots as image')
        action = imagemenu.addAction('Save &Layer Movie')
        action.triggered.connect(self._save_layer_movie)
        action.setToolTip('Save slices plot animation as a function of layer')
        action = imagemenu.addAction('Save &Iteration Movie')
        action.triggered.connect(self._save_iter_movie)
        action.setToolTip('Save slices plot animation as a function of iteration')

        # -- Color map picker
        cmapmenu = imagemenu.addMenu('&Color Map')
        self.color_map = QtWidgets.QActionGroup(self)
        self.color_map.setExclusive(True)
        starting_cmap = self.settings.value('cmap', defaultValue='coolwarm')
        for i, cmap in enumerate(['coolwarm', 'cubehelix', 'CMRmap', 'gray', 'gray_r', 'jet']):
            action = self.color_map.addAction(QtWidgets.QAction(cmap, self, checkable=True))
            if cmap == starting_cmap:
                action.setChecked(True)
            action.triggered.connect(self._cmap_changed)
            action.setToolTip('Set color map')
            cmapmenu.addAction(action)

        # Analysis menu
        analysismenu = menubar.addMenu('&Analysis')
        action = analysismenu.addAction('Open &Frameviewer')
        action.triggered.connect(self._open_frameviewer)
        action.setToolTip('View frames related to given mode')
        if self.recon_type == '3d':
            action.setEnabled(False)
        action = analysismenu.addAction('Subtract radial minimum')
        action.triggered.connect(self._subtract_radmin)
        action.setToolTip('Subtract radial minimum from intensities')

        modemenu = analysismenu.addMenu('Mode selection')
        action = modemenu.addAction('Toggle mode selection')
        action.setCheckable(True)
        action.setToolTip('Select modes from 2D classification')
        action.triggered.connect(self._toggle_mode_selection)
        self.blacklist_action = modemenu.addAction('Save blacklist file\n(%d good frames)'%self.num_good)
        self.blacklist_action.setToolTip('Save blacklist file with frames in selected modes')
        self.blacklist_action.triggered.connect(self._save_blacklist)

        action = analysismenu.addAction('Open &CLPCA')
        action.triggered.connect(self._open_clpca)
        action.setToolTip('Open CLPCA analysis window')

        action = analysismenu.addAction('2D Class &Phaser')
        action.triggered.connect(self._open_phaser2d)
        action.setToolTip('2D phasing of class averages')
        if self.recon_type == '3d':
            action.setEnabled(False)

    def _init_plotarea(self):
        plot_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        plot_splitter.setObjectName('plots')

        # Volume slices figure
        self.fig = matplotlib.figure.Figure(figsize=(14, 5))
        self.fig.subplots_adjust(left=0.0, bottom=0.00, right=0.99, wspace=0.0)
        #self.fig.set_facecolor('#232629')
        #self.fig.set_facecolor('#112244')
        self.fig.set_facecolor('#222222')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.show()
        plot_splitter.addWidget(self.canvas)
        self.vol_plotter = VolumePlotter(self.fig, self.recon_type, self.num_modes, self.num_nonrot, self.num_rot)
        self.vol_plotter.normvecs = self.settings.value('normvecs', defaultValue=np.identity(3))
        self.need_replot = self.vol_plotter.need_replot

        # Progress plots figure
        self.log_fig = matplotlib.figure.Figure(figsize=(14, 5), facecolor='w')
        #self.log_fig.set_facecolor('#232629')
        #self.log_fig.set_facecolor('#112244')
        self.log_fig.set_facecolor('#222222')
        self.plotcanvas = FigureCanvas(self.log_fig)
        self.plotcanvas.show()
        plot_splitter.addWidget(self.plotcanvas)
        self.log_plotter = LogPlotter(self.log_fig, self.folder)

        return plot_splitter

    def _init_optionsarea(self):
        options_widget = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout()
        options_widget.setLayout(vbox)

        # -- Log file
        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        label = QtWidgets.QLabel('Log file name:', self)
        hbox.addWidget(label)
        self.logfname = QtWidgets.QLineEdit(self.logfname, self)
        self.logfname.setMinimumWidth(160)
        self.logfname.setToolTip('Path to log file to get metrics and latest iterations')
        hbox.addWidget(self.logfname)
        label = QtWidgets.QLabel('VRange:', self)
        hbox.addWidget(label)
        vmin, vmax = self.settings.value('vrange', defaultValue=['0', '1'])
        self.rangemin = QtWidgets.QLineEdit(vmin, self)
        self.rangemin.setFixedWidth(48)
        self.rangemin.returnPressed.connect(self._range_changed)
        self.rangemin.setToolTip('Minimum value of color scale')
        hbox.addWidget(self.rangemin)
        self.rangestr = QtWidgets.QLineEdit(vmax, self)
        self.rangestr.setFixedWidth(48)
        self.rangestr.returnPressed.connect(self._range_changed)
        self.rangestr.setToolTip('Maximum value of color scale')
        hbox.addWidget(self.rangestr)

        # -- Volume file
        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        label = QtWidgets.QLabel('File name:', self)
        hbox.addWidget(label)
        if self.model_name is None:
            self.fname = QtWidgets.QLineEdit(self.folder+'/output_001.h5', self)
        else:
            self.fname = QtWidgets.QLineEdit(self.model_name, self)
        self.fname.setMinimumWidth(160)
        self.fname.setToolTip('Path to volume to be plotted')
        hbox.addWidget(self.fname)
        exponent = self.settings.value('exponent', defaultValue=1.0)
        label = QtWidgets.QLabel('Exp:', self)
        hbox.addWidget(label)
        self.expstr = QtWidgets.QLineEdit(str(exponent), self)
        self.expstr.setFixedWidth(48)
        self.expstr.returnPressed.connect(self._range_changed)
        self.expstr.setToolTip('Exponent, or gamma, for color scale. Enter the string "log" for the symlog normalization')
        hbox.addWidget(self.expstr)

        # -- Sliders
        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        label = QtWidgets.QLabel('Iteration', self)
        hbox.addWidget(label)
        self.iter_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.iter_slider.setRange(0, 1)
        self.iter_slider.sliderMoved.connect(self._iterslider_moved)
        self.iter_slider.sliderReleased.connect(self._iternum_changed)
        self.iter_slider.setToolTip('Set iteration to view')
        hbox.addWidget(self.iter_slider)
        self.iternum = MySpinBox(self)
        self.iternum.setValue(self.iter_slider.value())
        self.iternum.setMinimum(0)
        self.iternum.setMaximum(1)
        #self.iternum.valueChanged.connect(self._iternum_changed)
        self.iternum.editingFinished.connect(self._iternum_changed)
        self.iternum.setFixedWidth(60)
        self.iternum.setToolTip('Set iteration to view')
        hbox.addWidget(self.iternum)
        if self.recon_type == '3d':
            hbox = QtWidgets.QHBoxLayout()
            vbox.addLayout(hbox)
            label = QtWidgets.QLabel('Layer num.', self)
            hbox.addWidget(label)
            self.layer_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
            self.layer_slider.setRange(0, 200)
            self.layer_slider.sliderMoved.connect(self._layerslider_moved)
            self.layer_slider.sliderReleased.connect(self._layernum_changed)
            self.layer_slider.setToolTip('Set layer number in 3D volume')
            hbox.addWidget(self.layer_slider)
            self.layernum = MySpinBox(self)
            self.layernum.setValue(self.layer_slider.value())
            self.layernum.setMinimum(0)
            self.layernum.setMaximum(200)
            self.layernum.valueChanged.connect(self._layernum_changed)
            self.layernum.editingFinished.connect(self._layernum_changed)
            self.layernum.setFixedWidth(60)
            self.layernum.setToolTip('Set layer number in 3D volume')
            hbox.addWidget(self.layernum)
        if self.num_modes > 1:
            hbox = QtWidgets.QHBoxLayout()
            vbox.addLayout(hbox)
            label = QtWidgets.QLabel('Mode', self)
            hbox.addWidget(label)
            self.mode_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
            self.mode_slider.setRange(0, self.num_modes-1)
            self.mode_slider.sliderMoved.connect(self._modeslider_moved)
            self.mode_slider.sliderReleased.connect(self._modenum_changed)
            self.mode_slider.setToolTip('Set mode number')
            hbox.addWidget(self.mode_slider)
            self.modenum = MySpinBox(self)
            self.modenum.setValue(self.iter_slider.value())
            self.modenum.setMinimum(0)
            self.modenum.setMaximum(self.num_modes-1)
            #self.modenum.valueChanged.connect(self._modenum_changed)
            self.modenum.editingFinished.connect(self._modenum_changed)
            self.modenum.setFixedWidth(60)
            self.modenum.setToolTip('Set mode number')
            hbox.addWidget(self.modenum)
            self.old_modenum = self.modenum.value()

        # -- Buttons
        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        button = QtWidgets.QPushButton('Check', self)
        button.clicked.connect(self._check_for_new)
        button.setToolTip('Examine log file to see whether any new iterations have been completed')
        hbox.addWidget(button)
        self.ifcheck = QtWidgets.QCheckBox('Keep checking', self)
        self.ifcheck.stateChanged.connect(self._keep_checking)
        self.ifcheck.setChecked(False)
        self.ifcheck.setToolTip('Check log file every 5 seconds')
        hbox.addWidget(self.ifcheck)
        hbox.addStretch(1)
        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        hbox.addStretch(1)
        button = QtWidgets.QPushButton('Plot', self)
        button.clicked.connect(self._parse_and_plot)
        button.setToolTip('Plot volume (shortcut: ENTER)')
        hbox.addWidget(button)
        button = QtWidgets.QPushButton('Reparse', self)
        button.clicked.connect(lambda: self._parse_and_plot(force=True))
        button.setToolTip('Force reparsing of file and plot')
        hbox.addWidget(button)
        button = QtWidgets.QPushButton('Quit', self)
        button.clicked.connect(self.close)
        hbox.addWidget(button)

        # -- Log file display
        log_area = QtWidgets.QScrollArea(self)
        vbox.addWidget(log_area)
        log_area.setMinimumWidth(300)
        log_area.setWidgetResizable(True)
        self.emclog_text = QtWidgets.QTextEdit(
            'Press \'Check\' to synchronize with log file<br>'
            'Select \'Keep Checking\' to periodically synchronize<br><br>'
            'The top half of the display area will show three orthogonal<br>'
            'slices of the 3D volume. The bottom half will show plots of<br>'
            'various parameters vs iteration.', self)
        self.emclog_text.setReadOnly(True)
        self.emclog_text.setFontPointSize(8)
        self.emclog_text.setFontFamily('Courier')
        self.emclog_text.setFontWeight(QtGui.QFont.DemiBold)
        self.emclog_text.setTabStopWidth(22)
        self.emclog_text.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        self.emclog_text.setObjectName('logtext')
        self.emclog_text.setToolTip('Log file contents')
        log_area.setWidget(self.emclog_text)

        return options_widget

    def _refresh_gui(self):
        fpath = QtWidgets.QFileDialog.getOpenFileName(self, 'Load config file',
                                                      '.', 'Config file (*.ini)')
        if os.environ['QT_API'] == 'pyqt5':
            fname = fpath[0]
        else:
            fname = fpath
        if not fname:
            return
        self._save_settings()
        self.settings.endGroup()
        self._read_config(fname)
        self._init_main()

    def _layernum_changed(self, value=None):
        if value is None:
            # Slider released or editing finished
            self.need_replot = True
        elif value == self.layernum.value():
            self.layer_slider.setValue(value)
        self._parse_and_plot()

    def _layerslider_moved(self, value):
        self.layernum.setValue(int(value))

    def _iternum_changed(self, value=None):
        if value is None:
            self.fname.setText(self._gen_model_fname(self.iternum.value()))
        elif value == self.iternum.value():
            self.iter_slider.setValue(value)
            if self.need_replot:
                self.fname.setText(self._gen_model_fname(self.iternum.value()))
        self._parse_and_plot()

    def _iterslider_moved(self, value):
        self.iternum.setValue(value)

    def _modenum_changed(self, value=None):
        if value == self.modenum.value():
            self.mode_slider.setValue(value)
        if self.recon_type == '3d':
            self._parse_and_plot()
        else:
            self._plot_vol(update=True)

    def _modeslider_moved(self, value):
        self.modenum.setValue(value)

    def _range_changed(self):
        self.need_replot = True

    def _gen_model_fname(self, num):
        h5_fname = self.folder+'/output_%.3d.h5' % num
        if os.path.isfile(h5_fname):
            return h5_fname
        else:
            return self.folder+'/output/intens_%.3d.bin' % num

    def _read_config(self, config):
        self.settings.beginGroup(os.path.abspath(config).replace('/', '_'))
        try:
            self.folder = read_config.get_filename(config, 'emc', 'output_folder')
        except read_config.configparser.NoOptionError:
            self.folder = 'data/'

        try:
            self.logfname = read_config.get_filename(config, 'emc', 'log_file')
        except read_config.configparser.NoOptionError:
            self.logfname = 'EMC.log'

        try:
            self.recon_type = read_config.get_param(config, 'emc', 'recon_type').lower()
        except read_config.configparser.NoOptionError:
            self.recon_type = '3d'
        self.num_modes = 1
        self.num_nonrot = 0
        self.num_rot = None
        try:
            self.num_modes = int(read_config.get_param(config, 'emc', 'num_modes'))
            self.num_nonrot = int(read_config.get_param(config, 'emc', 'num_nonrot_modes'))
            self.num_rot = int(read_config.get_param(config, 'emc', 'num_rot'))
        except read_config.configparser.NoOptionError:
            pass
        self.config = config

    def _init_sliders(self, slider_type, numvals, init):
        if slider_type == 'layer':
            self.layer_slider.setRange(0, numvals-1)
            self.layernum.setMaximum(numvals-1)
            self.layer_slider.setValue(init)
            self._layerslider_moved(init)
        elif slider_type == 'mode':
            self.mode_slider.setRange(0, numvals-1)
            self.modenum.setMaximum(numvals-1)
            self.mode_slider.setValue(init)
            self._modeslider_moved(init)

    def _plot_vol(self, num=None, update=False):
        if self.recon_type == '2d':
            self.canvas.mpl_connect('button_press_event', self._select_mode)
            if num is None:
                if self.num_modes > 1:
                    num = int(self.modenum.text())
                else:
                    num = 0
        elif self.recon_type == '3d':
            self.canvas.mpl_connect('button_press_event', self._show_menu)
            self.canvas.mpl_connect('motion_notify_event', self._drag_normvec)
            if num is None:
                num = int(self.layernum.text())
        argsdict = {'vrange': (float(self.rangemin.text()), float(self.rangestr.text())),
                    'exponent': self.expstr.text(),
                    'cmap': self.color_map.checkedAction().text()}
        if update:
            self.vol_plotter.update_mode(num, **argsdict)
        else:
            self.vol_plotter.plot(num, **argsdict)
        if self.num_modes > 1:
            self.old_modenum = self.modenum.value()

    def _parse_and_plot(self, force=False, rots=True):
        if force or not self.vol_plotter.image_exists or self.old_fname != self.fname.text():
            if self.num_modes > 1:
                self._init_sliders('mode', self.num_modes+self.num_nonrot, self.modenum.value())
                modenum = self.modenum.value()
            else:
                modenum = 0
            self.old_fname, size, center = self.vol_plotter.parse(self.fname.text(),
                                            modenum=modenum, rots=rots)
            if self.recon_type == '3d':
                self._init_sliders('layer', size, center)
            self._plot_vol()
        elif self.num_modes > 1 and self.modenum.value() != self.old_modenum:
            self.old_fname, size, center = self.vol_plotter.parse(self.fname.text(),
                                             modenum=self.modenum.value(), rots=rots)
            if self.recon_type == '3d':
                self._init_sliders('layer', size, center)
            elif self.num_modes > 1:
                self._init_sliders('mode', self.num_modes+self.num_nonrot, self.modenum.value())
            self._plot_vol()
        elif self.need_replot:
            self._plot_vol()
        else:
            pass

    def _check_for_new(self):
        if not os.path.isfile(self.logfname.text()):
            return
        with open(self.logfname.text(), 'r') as fptr:
            last_line = fptr.readlines()[-1].rstrip().split()
        try:
            iteration = int(last_line[0])
        except ValueError:
            iteration = 0

        if iteration > 0 and self.max_iternum != iteration:
            self.fname.setText(self._gen_model_fname(iteration))
            self.max_iternum = iteration
            self.iter_slider.setRange(0, self.max_iternum)
            self.iternum.setMaximum(self.max_iternum)
            self.iter_slider.setValue(iteration)
            self._iterslider_moved(iteration)
            log_text = self.log_plotter.plot(self.logfname.text(),
                 self.color_map.checkedAction().text())
            self._parse_and_plot()
            self.emclog_text.setText(log_text)

    def _keep_checking(self):
        if self.ifcheck.isChecked():
            self._check_for_new()
            self.checker.timeout.connect(self._check_for_new)
            self.checker.start(5000)
        else:
            self.checker.stop()

    def _select_mode(self, event):
        curr_mode = -1
        for i, subp in enumerate(self.vol_plotter.subplot_list):
            if event.inaxes is subp:
                curr_mode = i
        if curr_mode < 0:
            return

        if curr_mode != self.modenum.value():
            self.mode_slider.setValue(curr_mode)
            self.modenum.setValue(curr_mode)
            self._modenum_changed()

            if self.fviewer is not None:
                self.fviewer.mode = curr_mode
                self.fviewer.label.setText('Class %d frames'%curr_mode)
                self.fviewer.numlist = np.where(self.vol_plotter.modes == curr_mode)[0]

        if self.mode_select:
            ax = self.vol_plotter.subplot_list[curr_mode]
            if len(ax.images) == 1:
                ax.imshow(np.zeros(self.vol_plotter.vol[curr_mode].shape), cmap='gray', alpha=0.5)
                self.num_good += (self.vol_plotter.modes == curr_mode).sum()
            else:
                ax.images[-1].remove()
                self.num_good -= (self.vol_plotter.modes == curr_mode).sum()
            self.blacklist_action.setText('Save blacklist file\n(%d good frames)'%self.num_good)
            self.vol_plotter.canvas.draw()

    def _show_menu(self, event):
        slice_num = -1
        for i, subp in enumerate(self.vol_plotter.subplot_list):
            if event.inaxes is subp:
                slice_num = i
        if slice_num == -1:
            return

        if event.button == 1 and 'ctrl' in event.modifiers:
            self.vol_plotter._init_yval = event.y
        if event.button == 1 and 'alt' in event.modifiers:
            self.vol_plotter._init_xval = event.x
        elif event.button == 3:
            context_menu = QtWidgets.QMenu()
            context_menu.addAction('Update normal vector', lambda:self._update_normvec(slice_num))
            cursor = QtGui.QCursor()
            context_menu.exec_(cursor.pos())

    def _update_normvec(self, slice_num):
        updater = NormVecUpdater(self.vol_plotter.normvecs[slice_num], self)
        if updater.vec is not None:
            self.vol_plotter.normvecs[slice_num] = updater.vec
            self.need_replot = True
            self._parse_and_plot()

    def _drag_normvec(self, event):
        if event.button != 1:
            return
        slice_num = -1
        for i, subp in enumerate(self.vol_plotter.subplot_list):
            if event.inaxes is subp:
                slice_num = i
        if slice_num == -1:
            return

        if 'ctrl' in event.modifiers:
            angle = np.sign(event.y-self.vol_plotter._init_yval) * np.pi / 36
            rotaxis = (slice_num+1)%3
        elif 'alt' in event.modifiers:
            angle = np.sign(event.x-self.vol_plotter._init_xval) * np.pi / 36
            rotaxis = (slice_num+2)%3
        else:
            return
        c = np.cos(angle)
        s = np.sin(angle)
        orig_vec = self.vol_plotter.normvecs[slice_num]
        mat = np.roll([[1,0,0],[0,c,-s],[0,s,c]], rotaxis, axis=(0,1))
        self.vol_plotter.normvecs[slice_num] = np.dot(mat, orig_vec)
        self.need_replot = True
        self._parse_and_plot()

    def _load_volume(self):
        fpath = QtWidgets.QFileDialog.getOpenFileName(self, 'Load 3D Volume',
                                                      'data/', 'Binary data (*.bin)')
        if os.environ['QT_API'] == 'pyqt5':
            fname = fpath[0]
        else:
            fname = fpath
        if fname:
            self.fname.setText(fname)
            self._parse_and_plot()

    def _save_plot(self):
        default_name = 'images/'+os.path.splitext(os.path.basename(self.fname.text()))[0]+'.png'
        fpath = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Volume Image',
                                                      default_name, 'Image (*.png)')
        if os.environ['QT_API'] == 'pyqt5':
            fname = fpath[0]
        else:
            fname = fpath
        if fname:
            self.fig.savefig(fname, bbox_inches='tight', dpi=120)
            sys.stderr.write('Saved to %s\n'%fname)

    def _save_log_plot(self):
        default_name = 'images/log_fig.png'
        fpath = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Log Plots',
                                                      default_name, 'Image (*.png)')
        if os.environ['QT_API'] == 'pyqt5':
            fname = fpath[0]
        else:
            fname = fpath
        if fname:
            self.log_fig.savefig(fname, bbox_inches='tight', dpi=120)
            sys.stderr.write("Saved to %s\n"%fname)

    def _plot_layer(self, num):
        self._plot_vol(num=num)
        self.fig.suptitle('Layer %d'%num, y=0.01, va='bottom')
        return self.fig,

    def _save_layer_movie(self):
        default_name = 'images/'+os.path.splitext(os.path.basename(self.fname.text()))[0]+'_layers.mp4'
        fpath = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Layer Animation Movie',
                                                      default_name, 'Movie (*.mp4)')
        if os.environ['QT_API'] == 'pyqt5':
            fname = fpath[0]
        else:
            fname = fpath
        if fname:
            sys.stderr.write('Saving layer animation to %s ...' % fname)
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=20, codec='h264', bitrate=1800)
            anim = animation.FuncAnimation(self.fig, self._plot_layer, self.layer_slider.maximum()+1, interval=50, repeat=False)
            anim.save(fname, writer=writer)
            self._parse_and_plot(force=True)
            sys.stderr.write('done\n')

    def _plot_iter(self, num):
        self.fname.setText(self._gen_model_fname(num))
        self._parse_and_plot()
        self.fig.suptitle('Iteration %d'%num, y=0.01, va='bottom')
        return self.fig,

    def _save_iter_movie(self):
        default_name = 'images/iterations.mp4'
        fpath = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Layer Animation Movie',
                                                      default_name, 'Movie (*.mp4)')
        if os.environ['QT_API'] == 'pyqt5':
            fname = fpath[0]
        else:
            fname = fpath
        if fname:
            sys.stderr.write('Saving iteration animation to %s ...' % fname)
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, codec='h264', bitrate=1800)
            anim = animation.FuncAnimation(self.fig, self._plot_iter, self.iter_slider.maximum()+1, interval=50, repeat=False)
            anim.save(fname, writer=writer)
            self._parse_and_plot(force=True)
            sys.stderr.write('done\n')

    def _cmap_changed(self):
        if self.vol_plotter.image_exists:
            self.need_replot = True
            self._parse_and_plot()

    def _open_frameviewer(self):
        if self.fviewer is not None:
            return
        if self.num_modes > 1 and self.vol_plotter.rots is not None:
            mode = self.modenum.value()
            numlist = np.where(self.vol_plotter.modes == mode)[0]
            self.fviewer = MyFrameviewer(self.config, mode, numlist)
        else:
            self.fviewer = MyFrameviewer(self.config, -1, [])
        self.fviewer.windowClosed.connect(self._fviewer_closed)

    def _open_clpca(self):
        if self.clpca is not None:
            return
        if self.vol_plotter.vol is None:
            print('Parse intensities first')
            return
        self.clpca = clpca.CLPCA(self)
        self.clpca.windowClosed.connect(self._clpca_closed)

    def _open_phaser2d(self):
        if self.phaser2d is not None:
            return
        if self.vol_plotter.vol is None:
            print('Parse intensities first')
            return
        self.phaser2d = phaser_gui.Phaser2D(self)
        self.phaser2d.windowClosed.connect(self._phaser2d_closed)

    def _subtract_radmin(self):
        self.vol_plotter.subtract_radmin()
        self._plot_vol()

    def _toggle_mode_selection(self, status):
        self.mode_select = status
        if not status:
            self.num_good = 0
            self.blacklist_action.setText('Save blacklist file\n(%d good frames)'%self.num_good)
            self.need_replot = True
            self._parse_and_plot()

    def _save_blacklist(self):
        good_modes = np.where([len(ax.images)>1 for ax in self.vol_plotter.subplot_list])[0]
        print('Good modes:', good_modes)
        blist = np.ones(self.vol_plotter.modes.shape, dtype='u1')
        for m in good_modes:
            blist[self.vol_plotter.modes == m] = 0
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Blacklist File', 'blacklist_%d.dat'%self.num_good)
        if fname != '':
            print('Saving', (blist==0).sum(), 'good frames to', fname)
            np.savetxt(fname, blist, fmt='%d')

    def _save_settings(self):
        self.settings.setValue('vrange', [self.rangemin.text(), self.rangestr.text()])
        self.settings.setValue('exponent', self.expstr.text())
        self.settings.setValue('cmap', self.color_map.checkedAction().text())
        self.settings.setValue('normvecs', self.vol_plotter.normvecs)

    @QtCore.Slot()
    def _fviewer_closed(self):
        self.fviewer = None

    @QtCore.Slot()
    def _clpca_closed(self):
        self.clpca = None

    @QtCore.Slot()
    def _phaser2d_closed(self):
        self.phaser2d = None

    def closeEvent(self, event): # pylint: disable=C0103
        if self.fviewer is not None:
            self.fviewer.close()
        self._save_settings()
        event.accept()

    def keyPressEvent(self, event): # pylint: disable=C0103
        '''Override of default keyPress event handler'''
        key = event.key()
        mod = int(event.modifiers())

        if key == QtCore.Qt.Key_Return or key == QtCore.Qt.Key_Enter:
            self._parse_and_plot()
        elif QtGui.QKeySequence(mod+key) == QtGui.QKeySequence('Ctrl+Q'):
            self.close()
        elif QtGui.QKeySequence(mod+key) == QtGui.QKeySequence('Ctrl+S'):
            self._save_plot()
        elif QtGui.QKeySequence(mod+key) == QtGui.QKeySequence('Ctrl+K'):
            self._check_for_new()
        else:
            event.ignore()

def main():
    '''Parses command line arguments and launches ProgressViewer'''
    parser = argparse.ArgumentParser(description='Dragonfly Progress Monitor')
    parser.add_argument('-c', '--config_file',
                        help='Path to config file. Default=config.ini', default='config.ini')
    parser.add_argument('-f', '--volume_file',
                        help='Show slices of particular file instead of output', default=None)
    args, unknown = parser.parse_known_args()

    app = QtWidgets.QApplication(unknown)
    pv = ProgressViewer(config=args.config_file, model=args.volume_file)
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
