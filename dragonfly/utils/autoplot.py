#!/usr/bin/env python

'''Module containing Dragonfly Progress Viewer'''

import sys
import os
import argparse
import numpy as np
from scipy import ndimage
import h5py
from PyQt5 import QtCore, QtWidgets, QtGui # pylint: disable=import-error
import matplotlib
matplotlib.use('qt5agg')
from matplotlib.backends.backend_qt5agg import FigureCanvas # pylint: disable=no-name-in-module
import matplotlib.pyplot as plt
import matplotlib.animation as animation
os.environ['QT_API'] = 'pyqt5'
from .autoplot_core import AutoplotController
from .autoplot_mpl import VolumePlotter, LogPlotter
from . import frameviewer
from .py_src import gui_utils
from .py_src import clpca
from .py_src import phaser_gui

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
        super(MyFrameviewer, self).__init__(config_file, do_compare=True, mask=True,
                                            noscroll=True, noplot=True)
        self.mode = mode
        self.numlist = numlist

        fp_layout = self.frame_panel.layout()
        fp_count = fp_layout.count()
        line = fp_layout.takeAt(fp_count-1)

        hbox = QtWidgets.QHBoxLayout()
        line.insertLayout(1, hbox)
        gui_utils.add_scroll_hbox(self, hbox)
        if self.blacklist is not None:
            self.frame_panel.skip_bad = QtWidgets.QCheckBox('Skip bad', self)
            self.frame_panel.skip_bad.setChecked(True)
            hbox.addWidget(self.frame_panel.skip_bad)
        if self.mode >= 0:
            self.label = QtWidgets.QLabel('Class %d frames'%self.mode)
            hbox.addWidget(self.label)
        fp_layout.addLayout(line)

        if self.blacklist is not None:
            self.frame_panel.good_ind = np.where(self.blacklist==0)[0]

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

class Viewer2D(QtWidgets.QMainWindow):
    windowClosed = QtCore.pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.output_fname = parent.fname.text()
        self.intens = parent.vol_plotter.vol
        self.curr_intens = None
        with h5py.File(self.output_fname, 'r') as f:
            occ = f['occupancies'][:]
            self.sel_occ = occ[occ.sum(1) > 0.5]

        self._init_ui()

    def _init_ui(self):
        if self.parent.css is not None:
            self.setStyleSheet(self.parent.css)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('2D Intensity Viewer')
        self.window = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout()
        self.window.setLayout(vbox)
        self.setCentralWidget(self.window)
        self.window.setObjectName('frame')

        self.fig = matplotlib.figure.Figure(figsize=(12, 6))
        self.fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.05)
        self.canvas = FigureCanvas(self.fig)
        self.navbar = gui_utils.MyNavigationToolbar(self.canvas, self)
        vbox.addWidget(self.navbar)
        vbox.addWidget(self.canvas, stretch=1)

        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Output file: %s'%self.output_fname, self)
        line.addWidget(label)
        label = QtWidgets.QLabel('(%d 2D averages)'%self.intens.shape[0], self)
        line.addWidget(label)
        line.addStretch(1)
        label = QtWidgets.QLabel('Filter size', self)
        line.addWidget(label)
        self.filt_size = QtWidgets.QLineEdit('1000', self)
        self.filt_size.editingFinished.connect(self._plot)
        line.addWidget(self.filt_size)
        label = QtWidgets.QLabel('Class', self)
        line.addWidget(label)
        self.class_num = QtWidgets.QSpinBox(self)
        self.class_num.setMinimum(0)
        self.class_num.setMaximum(self.intens.shape[0]-1)
        self.class_num.setValue(self.parent.modenum.value())
        self.class_num.valueChanged.connect(self._class_num_changed)
        line.addWidget(self.class_num)

        self._class_num_changed(self.class_num.value())
        self.show()

    def _class_num_changed(self, num):
        self.curr_intens = self.intens[num]
        self.curr_occ = self.sel_occ[:,num]
        self._plot()

    def _plot(self, state=None):
        try:
            filt_size = int(self.filt_size.text())
        except ValueError:
            print('Filter size needs to be an integer')
            return
        exponent = self.parent.expstr.text()
        rangemin = float(self.parent.rangemin.text())
        rangemax = float(self.parent.rangestr.text())
        if exponent == 'log':
            norm = matplotlib.colors.SymLogNorm(linthresh=rangemax*1.e-2, vmin=rangemin, vmax=rangemax)
        else:
            norm = matplotlib.colors.PowerNorm(float(exponent), vmin=rangemin, vmax=rangemax)
        cmap = self.parent.color_map.checkedAction().text()
        size = self.curr_intens.shape[-1]
        cen = size // 2
        plot_intens = self.curr_intens.copy()
        plot_intens[plot_intens<0] = np.nan

        try:
            ax = self.fig.get_axes()[0]
        except IndexError:
            ax = self.fig.add_subplot(121)
        for i in ax.images:
            i.remove()
        ax.imshow(plot_intens, extent=[-cen-0.5, cen+0.5, cen+0.5, -cen-0.5], norm=norm, cmap=cmap)
        ax.set_facecolor('dimgray')

        try:
            ax = self.fig.get_axes()[1]
            ax.lines[0].remove()
        except IndexError:
            ax = self.fig.add_subplot(122)
            ax.set_xlabel('Hit number', fontsize=12)
            ax.set_ylabel('Hit percent (smoothed)', fontsize=12)
            ax.set_title('Occupancy distribution', fontsize=14)
        ax.plot(ndimage.uniform_filter(self.curr_occ, filt_size)*100)

        self.canvas.draw()

    def closeEvent(self, event):
        self.windowClosed.emit()
        event.accept()

class ProgressViewer(QtWidgets.QMainWindow):
    '''GUI to track progress of EMC reconstruction
    Shows orthogonal volumes slices, plots of metrics vs iteration and log file
    Can periodically poll log file for updates and automatically update plots

    Can also be used to view slices through other 3D volumes using the '-f' option
    '''
    def __init__(self, config='config.ini', model=None):
        super(ProgressViewer, self).__init__()
        self.controller = AutoplotController(config=config, model=model)
        self.config = self.controller.config
        self.model_name = self.controller.model_name
        plt.style.use('dark_background')
        self.settings = QtCore.QSettings('DragonflyProgressViewer')

        self.beta_change = self.num_rot_change = []
        self.checker = QtCore.QTimer(self)

        self._sync_controller_state()
        self._init_ui()
        if model is not None:
            self._parse_and_plot(rots=False)
        self.controller.old_fname = self.fname.text()
        self.fviewer = None
        self.clpca = None
        self.phaser2d = None
        self.viewer2d = None

    def _sync_controller_state(self):
        self.output_parser = self.controller.output_parser
        self.logfname = self.controller.logfname
        self.folder = self.controller.folder
        self.recon_type = self.controller.recon_type
        self.num_modes = self.controller.num_modes
        self.num_nonrot = self.controller.num_nonrot
        self.num_rot = self.controller.num_rot
        self.config = self.controller.config
        self.model_name = self.controller.model_name

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

        self.controller.max_iternum = 0
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
        action = analysismenu.addAction('Subtract radial minimum')
        action.triggered.connect(self._subtract_radmin)
        action.setToolTip('Subtract radial minimum from intensities')
        if self.recon_type == '2d':
            action = analysismenu.addAction('Normalize high q')
            action.triggered.connect(self._normalize_highq)
            action.setToolTip('Normalize outer region for all classes to 1')
            action = analysismenu.addAction('Align models')
            action.triggered.connect(self._align_models)
            action.setToolTip('Align principal axes of all models')

        modemenu = analysismenu.addMenu('Mode selection')
        action = modemenu.addAction('Toggle mode selection')
        action.setCheckable(True)
        action.setToolTip('Select modes from 2D classification')
        action.triggered.connect(self._toggle_mode_selection)
        self.blacklist_action = modemenu.addAction('Save blacklist file\n(%d good frames)'%self.controller.num_good)
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
        self.vol_plotter = VolumePlotter(self.fig, self.output_parser)
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
        self.log_plotter = LogPlotter(self.log_fig, self.output_parser)

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
            self.fname = QtWidgets.QLineEdit(self.controller.get_initial_volume_fname(), self)
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
            self.controller.record_mode(self.modenum.value())

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
        return self.controller.gen_model_fname(num)

    def _read_config(self, config_fname):
        self.settings.beginGroup(os.path.abspath(config_fname).replace('/', '_'))
        self.controller.load_config(config_fname)
        self._sync_controller_state()

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
            self._cid = self.canvas.mpl_connect('button_press_event', self._select_mode)
            if num is None:
                if self.num_modes > 1:
                    num = int(self.modenum.text())
                else:
                    num = 0
        elif self.recon_type == '3d':
            self._cid = self.canvas.mpl_connect('button_press_event', self._show_menu)
            self._dragid = self.canvas.mpl_connect('motion_notify_event', self._drag_normvec)
            if num is None:
                num = int(self.layernum.text())
        argsdict = {'vrange': (float(self.rangemin.text()), float(self.rangestr.text())),
                    'exponent': self.expstr.text(),
                    'cmap': self.color_map.checkedAction().text()}
        if update:
            self.vol_plotter.update_mode(num, **argsdict)
        else:
            self.vol_plotter.plot(num, **argsdict)
            self._apply_mode_selection_overlays()
        if self.num_modes > 1:
            self.controller.record_mode(self.modenum.value())

    def _apply_mode_selection_overlays(self):
        if not self.controller.mode_select:
            return
        for mode in self.controller.selected_modes:
            if mode >= len(self.vol_plotter.subplot_list):
                continue
            ax = self.vol_plotter.subplot_list[mode]
            ax.imshow(np.zeros(self.vol_plotter.vol[mode].shape), cmap='gray', alpha=0.5)
        self.vol_plotter.canvas.draw()

    def _parse_and_plot(self, force=False, rots=True):
        current_modenum = self.modenum.value() if self.num_modes > 1 else 0
        action, modenum = self.controller.plan_plot(
            self.fname.text(), current_modenum, force=force,
            image_exists=self.vol_plotter.image_exists, need_replot=self.need_replot
        )
        if action == 'parse':
            if self.num_modes > 1:
                self._init_sliders('mode', self.num_modes+self.num_nonrot, modenum)
            fname, size, center = self.vol_plotter.parse(self.fname.text(), modenum=modenum, rots=rots)
            self.controller.record_parse(fname, modenum)
            if self.recon_type == '3d':
                self._init_sliders('layer', size, center)
            self._plot_vol()
        elif action == 'replot':
            self._plot_vol()

    def _check_for_new(self):
        update = self.controller.check_for_new_iteration(self.logfname.text())

        if update['updated']:
            self.fname.setText(update['fname'])
            self.iter_slider.setRange(0, self.controller.max_iternum)
            self.iternum.setMaximum(self.controller.max_iternum)
            self.iter_slider.setValue(update['iteration'])
            self._iterslider_moved(update['iteration'])
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
            if event.inaxes is self.vol_plotter.main_subp and event.button == 3:
                self._show_2dmenu(event)
            return

        if curr_mode != self.modenum.value():
            self.mode_slider.setValue(curr_mode)
            self.modenum.setValue(curr_mode)
            self._modenum_changed()

            if self.fviewer is not None:
                self.fviewer.mode = curr_mode
                self.fviewer.label.setText('Class %d frames'%curr_mode)
                self.fviewer.numlist = np.where(self.vol_plotter.modes == curr_mode)[0]

        if self.controller.mode_select:
            ax = self.vol_plotter.subplot_list[curr_mode]
            if self.controller.toggle_selected_mode(curr_mode, self.vol_plotter.modes):
                ax.imshow(np.zeros(self.vol_plotter.vol[curr_mode].shape), cmap='gray', alpha=0.5)
            else:
                ax.images[-1].remove()
            self.blacklist_action.setText('Save blacklist file\n(%d good frames)'%self.controller.num_good)
            self.vol_plotter.canvas.draw()

    def _show_2dmenu(self, event):
        context_menu = QtWidgets.QMenu()
        context_menu.addAction('View detailed', self._open_viewer2d)
        context_menu.addAction('2D Phaser', self._open_phaser2d)
        cursor = QtGui.QCursor()
        context_menu.exec_(cursor.pos())

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
                                                      'data/', 'Binary data (*.bin);;H5 output files (*.h5)')
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
        self.fviewer.frame_panel.iteration = self.iternum.value()
        self.fviewer.windowClosed.connect(self._fviewer_closed)
        self.fviewer._next_frame()

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

    def _open_viewer2d(self):
        if self.viewer2d is not None:
            return
        if self.vol_plotter.vol is None:
            print('Parse intensities first')
            return
        self.viewer2d = Viewer2D(self)
        self.viewer2d.windowClosed.connect(self._viewer2d_closed)
        pass

    def _subtract_radmin(self):
        self.vol_plotter.subtract_radmin()
        self._plot_vol()

    def _normalize_highq(self):
        self.vol_plotter.normalize_highq()
        self._plot_vol()

    def _align_models(self):
        self.vol_plotter.align_models()
        self._plot_vol()

    def _toggle_mode_selection(self, status):
        self.controller.set_mode_selection(status)
        if not status:
            self.blacklist_action.setText('Save blacklist file\n(%d good frames)'%self.controller.num_good)
            self.need_replot = True
            self._parse_and_plot()

    def _save_blacklist(self):
        print('Good modes:', sorted(self.controller.selected_modes))
        blist = self.controller.generate_blacklist(self.vol_plotter.modes)
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Blacklist File', 'blacklist_%d.dat'%self.controller.num_good)
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

    @QtCore.Slot()
    def _viewer2d_closed(self):
        self.viewer2d = None

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
