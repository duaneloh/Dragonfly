#!/usr/bin/env python

'''Module containing Dragonfly Progress Viewer'''

from __future__ import print_function
import sys
import os
import argparse
import numpy as np
try:
    from PyQt5 import QtCore, QtWidgets, QtGui # pylint: disable=import-error
    import matplotlib
    matplotlib.use('qt5agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvas # pylint: disable=no-name-in-module
    import matplotlib.pyplot as plt
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
    os.environ['QT_API'] = 'pyqt'
from py_src import read_config

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

class VolumePlotter(object):
    def __init__(self, fig, recon_type='3d', num_modes=1):
        self.fig = fig
        self.canvas = fig.canvas
        self.vol = None
        self.old_modenum = None
        self.recon_type = recon_type
        self.num_modes = num_modes
        self.need_replot = False
        self.image_exists = False

    def parse(self, fname, modenum=0):
        '''Parse volume defined in options panel
        Can be either 3D volume or 2d slice stack depending on mode in config file
        '''
        if os.path.isfile(fname):
            self.vol = np.fromfile(fname, dtype='f8')
        else:
            sys.stderr.write("Unable to open %s\n"%fname)
            return 0, 0

        if self.recon_type == '3d':
            size = int(np.ceil(np.power(len(self.vol)/self.num_modes, 1./3.)))
            if self.num_modes > 1:
                self.vol = self.vol[modenum*size**3:(modenum+1)*size**3].reshape(size, size, size)
            else:
                self.vol = self.vol.reshape(size, size, size)
            center = size/2
            return_val = (fname, size, center)
        else:
            size = int(np.ceil(np.power(len(self.vol)/self.num_modes, 1./2.)))
            self.vol = self.vol.reshape(self.num_modes, size, size)
            center = 0
            return_val = (fname, self.num_modes, center)

        self.old_fname = fname
        if self.num_modes > 1:
            self.old_modenum = modenum
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
        self.fig.clf()

        if self.recon_type == '3d':
            subp = self.fig.add_subplot(131)
            vslice = self.vol[num, :, :]**exponent
            subp.imshow(vslice, vmin=rangemin, vmax=rangemax, cmap=cmap, interpolation='none')
            subp.set_title("YZ plane", y=1.01)
            subp.axis('off')

            subp = self.fig.add_subplot(132)
            vslice = self.vol[:, num, :]**exponent
            subp.matshow(vslice, vmin=rangemin, vmax=rangemax, cmap=cmap, interpolation='none')
            subp.set_title("XZ plane", y=1.01)
            subp.axis('off')

            subp = self.fig.add_subplot(133)
            vslice = self.vol[:, :, num]**exponent
            subp.matshow(vslice, vmin=rangemin, vmax=rangemax, cmap=cmap, interpolation='none')
            subp.set_title("XY plane", y=1.01)
            subp.axis('off')
        elif self.recon_type == '2d':
            numx = int(np.ceil(2.*np.sqrt(self.num_modes / 2.)))
            numy = int(np.ceil(self.num_modes / float(numx)))
            total_numx = numx + int(np.ceil(numx / 2)) + 1

            gspec = matplotlib.gridspec.GridSpec(numy, total_numx)
            gspec.update(wspace=0.02, hspace=0.02)
            self.subplot_list = []
            for mode in range(self.num_modes):
                subp = self.fig.add_subplot(gspec[mode//numx, mode%numx])
                subp.imshow(self.vol[mode]**exponent, vmin=rangemin, vmax=rangemax,
                            cmap=cmap, interpolation='none')
                #subp.text(0.05, 0.85, '%d'%mode, transform=subp.transAxes, fontsize=10,
                #          color='w', bbox={'facecolor': 'black', 'pad': 0})
                subp.text(0.05, 0.85, '%d'%mode, transform=subp.transAxes, fontsize=10, color='w')
                subp.axis('off')
                self.subplot_list.append(subp)
            subp = self.fig.add_subplot(gspec[:, numx:])
            subp.imshow(self.vol[num]**exponent, vmin=rangemin, vmax=rangemax,
                        cmap=cmap, interpolation='none')
            subp.set_title('Class %d'%num)
            subp.axis('off')

        self.canvas.draw()
        self.image_exists = True
        self.need_replot = False

class LogPlotter(object):
    def __init__(self, fig, folder='data/'):
        self.fig = fig
        self.canvas = fig.canvas
        self.folder = folder

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

        # Read orientation files for the first n iterations
        orient = []
        for i in range(len(loglines)):
            fname = self.folder+'/orientations/orientations_%.3d.bin' % (i+1)
            with open(fname, 'r') as fptr:
                orient.append(np.fromfile(fptr, '=i4'))
        olengths = np.array([len(ori) for ori in orient])
        max_length = olengths.max()

        iternum = loglines[:, 0].astype('i4')
        num_rot = loglines[:, 5].astype('i4')
        beta = loglines[:, 6].astype('f8')
        self.num_rot_change = np.append(np.where(np.diff(num_rot) != 0)[0], num_rot.shape[0])
        self.beta_change = np.where(np.diff(beta) != 0.)[0]

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

        self.fig.clf()
        grid = matplotlib.gridspec.GridSpec(2, 3, wspace=0.3, hspace=0.2)
        grid.update(left=0.05, right=0.99, hspace=0.2, wspace=0.3)

        self._add_logplot(grid[:, 0], iternum, loglines[:, 2],
                          'RMS change')
        self._add_logplot(grid[0, 1], iternum, loglines[:, 3],
                          r'Mutual info. $I(K,\Omega | W)$')
        self._add_logplot(grid[1, 1], iternum, loglines[:, 4],
                          'Avg log-likelihood', yscale='symlog')

        # Plot most likely orientation convergence plot
        if len(loglines) > 1:
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

    def _add_logplot(self, gridpos, xval, yval, title='', yscale='log'):
        subp = self.fig.add_subplot(gridpos)
        subp.plot(xval, yval.astype('f8'), 'o-')
        subp.set_yscale(yscale)
        subp.set_xlabel('Iteration')
        subp.set_ylabel(title)
        ylim = subp.get_ylim()
        subp.set_ylim(ylim)
        for i in self.beta_change:
            subp.plot([i+1, i+1], ylim, 'w--', lw=1)
        for i in self.num_rot_change[:-1]:
            subp.plot([i+1, i+1], ylim, 'r--', color='tab:orange', lw=1)

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
        self.max_iternum = 0
        plt.style.use('dark_background')

        self.beta_change = self.num_rot_change = []
        self.checker = QtCore.QTimer(self)

        self._read_config(config)
        self._init_ui()
        if model is not None:
            self._parse_and_plot()
        self.old_fname = self.fname.text()

    def _init_ui(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'py_src/style.css'), 'r') as f:
            self.setStyleSheet(f.read())
        self.setWindowTitle('Dragonfly Progress Viewer')
        self.setGeometry(100, 100, 1600, 800)
        overall = QtWidgets.QWidget()
        self.setCentralWidget(overall)
        layout = QtWidgets.QHBoxLayout(overall)
        layout.setContentsMargins(0, 0, 0, 0)

        self._init_menubar()
        plot_splitter = self._init_plotarea()
        options_widget = self._init_optionsarea()

        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_splitter.setObjectName('frame')
        layout.addWidget(main_splitter)
        main_splitter.addWidget(plot_splitter)
        main_splitter.addWidget(options_widget)

        self.show()

    def _init_menubar(self):
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)

        # File Menu
        filemenu = menubar.addMenu('&File')
        action = QtWidgets.QAction('&Load Volume', self)
        action.triggered.connect(self._load_volume)
        filemenu.addAction(action)
        action = QtWidgets.QAction('&Save Image', self)
        action.triggered.connect(self._save_plot)
        filemenu.addAction(action)
        action = QtWidgets.QAction('Save Log &Plot', self)
        action.triggered.connect(self._save_log_plot)
        filemenu.addAction(action)
        action = QtWidgets.QAction('&Quit', self)
        action.triggered.connect(self.close)
        filemenu.addAction(action)

        # Color map picker
        cmapmenu = menubar.addMenu('&Color Map')
        self.color_map = QtWidgets.QActionGroup(self, exclusive=True)
        for i, cmap in enumerate(['coolwarm', 'cubehelix', 'CMRmap', 'gray', 'gray_r', 'jet']):
            action = self.color_map.addAction(QtWidgets.QAction(cmap, self, checkable=True))
            if i == 0:
                action.setChecked(True)
            action.triggered.connect(self._cmap_changed)
            cmapmenu.addAction(action)

    def _init_plotarea(self):
        plot_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        plot_splitter.setObjectName('plots')

        # Volume slices figure
        self.fig = matplotlib.figure.Figure(figsize=(14, 5))
        self.fig.subplots_adjust(left=0.0, bottom=0.00, right=0.99, wspace=0.0)
        #self.fig.set_facecolor('#232629')
        self.fig.set_facecolor('#112244')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.show()
        plot_splitter.addWidget(self.canvas)
        self.vol_plotter = VolumePlotter(self.fig, self.recon_type, self.num_modes)
        self.need_replot = self.vol_plotter.need_replot

        # Progress plots figure
        self.log_fig = matplotlib.figure.Figure(figsize=(14, 5), facecolor='w')
        #self.log_fig.set_facecolor('#232629')
        self.log_fig.set_facecolor('#112244')
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
        hbox.addWidget(self.logfname)
        label = QtWidgets.QLabel('VRange:', self)
        hbox.addWidget(label)
        self.rangemin = QtWidgets.QLineEdit('0', self)
        self.rangemin.setFixedWidth(48)
        self.rangemin.returnPressed.connect(self._range_changed)
        hbox.addWidget(self.rangemin)
        self.rangestr = QtWidgets.QLineEdit('1', self)
        self.rangestr.setFixedWidth(48)
        self.rangestr.returnPressed.connect(self._range_changed)
        hbox.addWidget(self.rangestr)

        # -- Volume file
        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        label = QtWidgets.QLabel('File name:', self)
        hbox.addWidget(label)
        if self.model_name is None:
            self.fname = QtWidgets.QLineEdit(self.folder+'/output/intens_001.bin', self)
        else:
            self.fname = QtWidgets.QLineEdit(self.model_name, self)
        self.fname.setMinimumWidth(160)
        hbox.addWidget(self.fname)
        label = QtWidgets.QLabel('Exp:', self)
        hbox.addWidget(label)
        self.expstr = QtWidgets.QLineEdit('1', self)
        self.expstr.setFixedWidth(48)
        self.expstr.returnPressed.connect(self._range_changed)
        hbox.addWidget(self.expstr)

        # -- Sliders
        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        label = QtWidgets.QLabel('Layer num.', self)
        hbox.addWidget(label)
        self.layer_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.layer_slider.setRange(0, 200)
        self.layer_slider.sliderMoved.connect(self._layerslider_moved)
        self.layer_slider.sliderReleased.connect(self._layernum_changed)
        hbox.addWidget(self.layer_slider)
        self.layernum = MySpinBox(self)
        self.layernum.setValue(self.layer_slider.value())
        self.layernum.setMinimum(0)
        self.layernum.setMaximum(200)
        self.layernum.valueChanged.connect(self._layernum_changed)
        self.layernum.editingFinished.connect(self._layernum_changed)
        self.layernum.setFixedWidth(48)
        hbox.addWidget(self.layernum)
        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        label = QtWidgets.QLabel('Iteration', self)
        hbox.addWidget(label)
        self.iter_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.iter_slider.setRange(0, 1)
        self.iter_slider.sliderMoved.connect(self._iterslider_moved)
        self.iter_slider.sliderReleased.connect(self._iternum_changed)
        hbox.addWidget(self.iter_slider)
        self.iternum = MySpinBox(self)
        self.iternum.setValue(self.iter_slider.value())
        self.iternum.setMinimum(0)
        self.iternum.setMaximum(1)
        self.iternum.valueChanged.connect(self._iternum_changed)
        self.iternum.editingFinished.connect(self._iternum_changed)
        self.iternum.setFixedWidth(48)
        hbox.addWidget(self.iternum)
        if self.num_modes > 1:
            hbox = QtWidgets.QHBoxLayout()
            vbox.addLayout(hbox)
            label = QtWidgets.QLabel('Mode', self)
            hbox.addWidget(label)
            self.mode_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
            self.mode_slider.setRange(0, self.num_modes-1)
            self.mode_slider.sliderMoved.connect(self._modeslider_moved)
            self.mode_slider.sliderReleased.connect(self._modenum_changed)
            hbox.addWidget(self.mode_slider)
            self.modenum = MySpinBox(self)
            self.modenum.setValue(self.iter_slider.value())
            self.modenum.setMinimum(0)
            self.modenum.setMaximum(self.num_modes-1)
            self.modenum.valueChanged.connect(self._modenum_changed)
            self.modenum.editingFinished.connect(self._modenum_changed)
            self.modenum.setFixedWidth(48)
            hbox.addWidget(self.modenum)
            self.old_modenum = self.modenum.value()

        # -- Buttons
        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        button = QtWidgets.QPushButton('Check', self)
        button.clicked.connect(self._check_for_new)
        hbox.addWidget(button)
        self.ifcheck = QtWidgets.QCheckBox('Keep checking', self)
        self.ifcheck.stateChanged.connect(self._keep_checking)
        self.ifcheck.setChecked(False)
        hbox.addWidget(self.ifcheck)
        hbox.addStretch(1)
        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        hbox.addStretch(1)
        button = QtWidgets.QPushButton('Plot', self)
        button.clicked.connect(self._parse_and_plot)
        hbox.addWidget(button)
        button = QtWidgets.QPushButton('Reparse', self)
        button.clicked.connect(self._force_plot)
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
        log_area.setWidget(self.emclog_text)

        return options_widget

    def _layernum_changed(self, value=None):
        if value is None:
            # Slider released or editing finished
            self.need_replot = True
        elif value == self.layernum.value():
            self.layer_slider.setValue(value)
        self._parse_and_plot()

    def _layerslider_moved(self, value):
        self.layernum.setValue(value)

    def _iternum_changed(self, value=None):
        if value is None:
            self.fname.setText(self.folder+'/output/intens_%.3d.bin' % self.iternum.value())
        elif value == self.iternum.value():
            self.iter_slider.setValue(value)
            if self.need_replot:
                self.fname.setText(self.folder+'/output/intens_%.3d.bin' % value)
        self._parse_and_plot()

    def _iterslider_moved(self, value):
        self.iternum.setValue(value)

    def _modenum_changed(self, value=None):
        if value == self.modenum.value():
            self.mode_slider.setValue(value)
        self._parse_and_plot()

    def _modeslider_moved(self, value):
        self.modenum.setValue(value)

    def _range_changed(self):
        self.need_replot = True

    def _read_config(self, config):
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
        try:
            self.num_modes = int(read_config.get_param(config, 'emc', 'num_modes'))
        except read_config.configparser.NoOptionError:
            self.num_modes = 1

    def _update_layers(self, size, center):
        self.layer_slider.setRange(0, size-1)
        self.layernum.setMaximum(size-1)
        self.layer_slider.setValue(center)
        self._layerslider_moved(center)

    def _plot_vol(self, num=None):
        if num is None:
            num = int(self.layernum.text())
        self.vol_plotter.plot(num,
                              (float(self.rangemin.text()), 
                               float(self.rangestr.text())),
                              float(self.expstr.text()),
                              self.color_map.checkedAction().text())
        if self.recon_type == '2d':
            self.canvas.mpl_connect('button_press_event', self._select_mode)

    def _parse_and_plot(self):
        if not self.vol_plotter.image_exists or self.old_fname != self.fname.text():
            self.old_fname, size, center = self.vol_plotter.parse(self.fname.text())
            self._update_layers(size, center)
            self._plot_vol()
        elif self.num_modes > 1 and self.modenum.value() != self.old_modenum:
            self.old_fname, size, center = self.vol_plotter.parse(self.fname.text(),
                                             modenum=self.modenum.value())
            self._update_layers(size, center)
            self._plot_vol()
        elif self.need_replot:
            self._plot_vol()
        else:
            pass

    def _check_for_new(self):
        with open(self.logfname.text(), 'r') as fptr:
            last_line = fptr.readlines()[-1].rstrip().split()
        try:
            iteration = int(last_line[0])
        except ValueError:
            iteration = 0

        if iteration > 0 and self.max_iternum != iteration:
            self.fname.setText(self.folder+'/output/intens_%.3d.bin' % iteration)
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
        if curr_mode >= 0 and curr_mode != self.layernum.value():
            self.layer_slider.setValue(curr_mode)
            self.layernum.setValue(curr_mode)
            self._plot_vol(curr_mode)

    def _force_plot(self):
        self.old_fname, size, center = self.vol_plotter.parse(self.fname.text())
        self._update_layers(size, center)
        self._plot_vol()

    def _load_volume(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load 3D Volume',
                                                         'data/', 'Binary data (*.bin)')
        if fname:
            self.fname.setText(fname)
            self._parse_and_plot()

    def _save_plot(self):
        default_name = 'images/'+os.path.splitext(os.path.basename(self.fname.text()))[0]+'.png'
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Volume Image',
                                                         default_name, 'Image (*.png)')
        if fname:
            self.fig.savefig(fname, bbox_inches='tight', dpi=120)
            sys.stderr.write('Saved to %s\n'%fname)

    def _save_log_plot(self):
        default_name = 'images/log_fig.png'
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Log Plots',
                                                         default_name, 'Image (*.png)')
        if fname:
            self.log_fig.savefig(fname, bbox_inches='tight', dpi=120)
            sys.stderr.write("Saved to %s\n"%fname)

    def _cmap_changed(self):
        if self.vol_plotter.image_exists:
            self.need_replot = True
            self._parse_and_plot()

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
    ProgressViewer(config=args.config_file, model=args.volume_file)
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
