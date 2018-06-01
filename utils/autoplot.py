#!/usr/bin/env python

import sys
import os
import glob
import argparse
import numpy as np
import matplotlib
try:
    from PyQt5 import QtCore, QtWidgets, QtGui
    matplotlib.use('qt5agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvas
    os.environ['QT_API'] = 'pyqt5'
except ImportError:
    import sip; sip.setapi('QString', 2)
    from PyQt4 import QtCore, QtGui
    from PyQt4 import QtGui as QtWidgets
    matplotlib.use('qt4agg')
    from matplotlib.backends.backend_qt4agg import FigureCanvas
    os.environ['QT_API'] = 'pyqt'
import qdarkstyle
import matplotlib.pyplot as plt
from py_src import read_config

class mySpinBox(QtWidgets.QSpinBox):
    def __init__(self, parent, *args, **kwargs):
        super(mySpinBox, self).__init__(parent, *args, **kwargs)
        self.parent = parent

    def stepBy(self, steps):
        target_value = self.value() + steps
        if (target_value < self.minimum()):
            self.setValue(self.minimum())
        elif (target_value > self.maximum()):
            self.setValue(self.maximum())
        else:
            self.setValue(target_value)
        self.parent.need_replot = True

class Progress_viewer(QtWidgets.QMainWindow):
    def __init__(self, config='config.ini', model=None):
        super(Progress_viewer, self).__init__()
        self.config = config
        self.model_name = model
        self.log_txt = ""
        self.max_iternum = 0
        self.need_replot = False
        self.image_exists = False
        plt.style.use('dark_background')
        '''
        matplotlib.rcParams.update({
            'text.color': '#eff0f1',
            'xtick.color': '#eff0f1',
            'ytick.color': '#eff0f1',
            'axes.facecolor': '#eff0f1',
            'axes.labelcolor': '#eff0f1'})
        '''

        self.read_config(config)
        self.init_UI()
        if model is not None:
            self.parse_and_plot()
        self.old_fname = self.fname.text()

    def init_UI(self):
        self.setWindowTitle('Dragonfly Progress Viewer')
        self.setGeometry(100, 100, 1600, 800)
        overall = QtWidgets.QWidget()
        self.setCentralWidget(overall)
        
        layout = QtWidgets.QHBoxLayout(overall)
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(main_splitter)
        plot_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        main_splitter.addWidget(plot_splitter)
        
        # Menu items
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        
        # File Menu
        filemenu = menubar.addMenu('&File')
        action = QtWidgets.QAction('&Load Volume', self)
        action.triggered.connect(self.load_volume)
        filemenu.addAction(action)
        action = QtWidgets.QAction('&Save Image', self)
        action.triggered.connect(self.save_plot)
        filemenu.addAction(action)
        action = QtWidgets.QAction('Save Log &Plot', self)
        action.triggered.connect(self.save_log_plot)
        filemenu.addAction(action)
        action = QtWidgets.QAction('&Quit', self)
        action.triggered.connect(self.close)
        filemenu.addAction(action)
        
        # Color map picker
        cmapmenu = menubar.addMenu('&Color Map')
        self.color_map = QtWidgets.QActionGroup(self, exclusive=True)
        for i, s in enumerate(['coolwarm', 'cubehelix', 'CMRmap', 'gray', 'gray_r', 'jet']):
            a = self.color_map.addAction(QtWidgets.QAction(s, self, checkable=True))
            if i == 0:
                a.setChecked(True)
            a.triggered.connect(self.cmap_changed)
            cmapmenu.addAction(a)

        # Volume slices figure
        self.fig = matplotlib.figure.Figure(figsize=(14,5))
        self.fig.subplots_adjust(left=0.0, bottom=0.00, right=0.99, wspace=0.0)
        self.fig.set_facecolor('#232629')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.show()
        plot_splitter.addWidget(self.canvas)

        # Progress plots figure
        self.log_fig = matplotlib.figure.Figure(figsize=(14,5), facecolor='w')
        self.log_fig.set_facecolor('#232629')
        self.plotcanvas = FigureCanvas(self.log_fig)
        self.plotcanvas.show()
        plot_splitter.addWidget(self.plotcanvas)

        # Plot options widget
        options_widget = QtWidgets.QWidget()
        self.options = QtWidgets.QVBoxLayout()
        options_widget.setLayout(self.options)
        main_splitter.addWidget(options_widget)

        # -- Log file
        hbox = QtWidgets.QHBoxLayout()
        self.options.addLayout(hbox)
        label = QtWidgets.QLabel('Log file name:', self)
        hbox.addWidget(label)
        self.logfname = QtWidgets.QLineEdit(self.logfname, self)
        self.logfname.setMinimumWidth(160)
        hbox.addWidget(self.logfname)
        label = QtWidgets.QLabel('PlotMin:', self)
        hbox.addWidget(label)
        self.rangemin = QtWidgets.QLineEdit('0', self)
        self.rangemin.setFixedWidth(48)
        self.rangemin.returnPressed.connect(self.range_changed)
        hbox.addWidget(self.rangemin)
        label = QtWidgets.QLabel('PlotMax:', self)
        hbox.addWidget(label)
        self.rangestr = QtWidgets.QLineEdit('1', self)
        self.rangestr.setFixedWidth(48)
        self.rangestr.returnPressed.connect(self.range_changed)
        hbox.addWidget(self.rangestr)

        # -- Volume file
        hbox = QtWidgets.QHBoxLayout()
        self.options.addLayout(hbox)
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
        self.expstr.returnPressed.connect(self.range_changed)
        hbox.addWidget(self.expstr)

        # -- Sliders
        hbox = QtWidgets.QHBoxLayout()
        self.options.addLayout(hbox)
        label = QtWidgets.QLabel('Layer num.', self)
        hbox.addWidget(label)
        self.layer_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.layer_slider.setRange(0, 200)
        self.layer_slider.sliderMoved.connect(self.layerslider_moved)
        self.layer_slider.sliderReleased.connect(self.layernum_changed)
        hbox.addWidget(self.layer_slider)
        self.layernum = mySpinBox(self)
        self.layernum.setValue(self.layer_slider.value())
        self.layernum.setMinimum(0)
        self.layernum.setMaximum(200)
        self.layernum.valueChanged.connect(self.layernum_changed)
        self.layernum.editingFinished.connect(self.layernum_changed)
        self.layernum.setFixedWidth(48)
        hbox.addWidget(self.layernum)
        hbox = QtWidgets.QHBoxLayout()
        self.options.addLayout(hbox)
        label = QtWidgets.QLabel('Iteration', self)
        hbox.addWidget(label)
        self.iter_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.iter_slider.setRange(0, 1)
        self.iter_slider.sliderMoved.connect(self.iterslider_moved)
        self.iter_slider.sliderReleased.connect(self.iternum_changed)
        hbox.addWidget(self.iter_slider)
        self.iternum = mySpinBox(self)
        self.iternum.setValue(self.iter_slider.value())
        self.iternum.setMinimum(0)
        self.iternum.setMaximum(1)
        self.iternum.valueChanged.connect(self.iternum_changed)
        self.iternum.editingFinished.connect(self.iternum_changed)
        self.iternum.setFixedWidth(48)
        hbox.addWidget(self.iternum)

        # -- Buttons
        hbox = QtWidgets.QHBoxLayout()
        self.options.addLayout(hbox)
        button = QtWidgets.QPushButton('Check', self)
        button.clicked.connect(self.check_for_new)
        hbox.addWidget(button)
        self.ifcheck = QtWidgets.QCheckBox('Keep checking', self)
        self.ifcheck.stateChanged.connect(self.keep_checking)
        self.ifcheck.setChecked(False)
        hbox.addWidget(self.ifcheck)
        hbox.addStretch(1)
        hbox = QtWidgets.QHBoxLayout()
        self.options.addLayout(hbox)
        hbox.addStretch(1)
        button = QtWidgets.QPushButton('Plot', self)
        button.clicked.connect(self.parse_and_plot)
        hbox.addWidget(button)
        button = QtWidgets.QPushButton('Reparse', self)
        button.clicked.connect(self.force_plot)
        hbox.addWidget(button)
        button = QtWidgets.QPushButton('Quit', self)
        button.clicked.connect(self.close)
        hbox.addWidget(button)

        # -- Log file display
        log_area = QtWidgets.QScrollArea(self)
        self.options.addWidget(log_area)
        log_area.setMinimumWidth(450)
        log_area.setWidgetResizable(True)
        self.emclog_text = QtWidgets.QTextEdit(
                'Press \'Check\' to synchronize with log file<br>'
                'Select \'Keep Checking\' to periodically synchronize<br><br>'
                'The top half of the display area will show three orthogonal slices of the 3D volume. '
                'The bottom half will show plots of various parameters vs iteration.', self)
        self.emclog_text.setReadOnly(True)
        self.emclog_text.setFontPointSize(8)
        self.emclog_text.setFontFamily('Courier')
        self.emclog_text.setFontWeight(QtGui.QFont.DemiBold)
        self.emclog_text.setTabStopWidth(22)
        log_area.setWidget(self.emclog_text)

        self.show()

    def layernum_changed(self, value=None, replot=True):
        if value is None:
            # Slider released or editing finished
            self.need_replot = True
        elif value == self.layernum.value():
            self.layer_slider.setValue(value)
        self.parse_and_plot()

    def layerslider_moved(self, value):
        self.layernum.setValue(value)
        
    def iternum_changed(self, value=None):
        if value is None:
            self.fname.setText(self.folder+'/output/intens_%.3d.bin' % self.iternum.value())
        elif value == self.iternum.value():
            self.iter_slider.setValue(value)
            if self.need_replot:
                self.fname.setText(self.folder+'/output/intens_%.3d.bin' % value)
        self.parse_and_plot()

    def iterslider_moved(self, value):
        self.iternum.setValue(value)

    def range_changed(self):
        self.need_replot = True

    def read_config(self, config):
        try:
            self.folder = read_config.get_filename(config, 'emc', 'output_folder')
        except read_config.ConfigParser.NoOptionError:
            self.folder = 'data/'

        try:
            self.logfname = read_config.get_filename(config, 'emc', 'log_file')
        except read_config.ConfigParser.NoOptionError:
            self.logfname = 'EMC.log'

        try:
            self.num_modes = int(read_config.get_param(config, 'emc', 'num_modes'))
        except read_config.ConfigParser.NoOptionError:
            self.num_modes = 0

    def plot_vol(self, num):
        #self.imagename.setText('images/' + os.path.splitext(os.path.basename(self.fname.text()))[0] + '.png')
        rangemin = float(self.rangemin.text())
        rangemax = float(self.rangestr.text())
        exponent = float(self.expstr.text())
        cmap = self.color_map.checkedAction().text()

        self.fig.clf()

        if self.num_modes == 0:
            a = self.vol[num,:,:]**exponent
            b = self.vol[:,num,:]**exponent
            c = self.vol[:,:,num]**exponent

            s1 = self.fig.add_subplot(131)
            s1.imshow(a, vmin=rangemin, vmax=rangemax, cmap=cmap, interpolation='none')
            s1.set_title("YZ plane", y=1.01)
            s1.axis('off')
            s2 = self.fig.add_subplot(132)
            s2.matshow(b, vmin=rangemin, vmax=rangemax, cmap=cmap, interpolation='none')
            s2.set_title("XZ plane", y=1.01)
            s2.axis('off')
            s3 = self.fig.add_subplot(133)
            s3.matshow(c, vmin=rangemin, vmax=rangemax, cmap=cmap, interpolation='none')
            s3.set_title("XY plane", y=1.01)
            s3.axis('off')
        else:
            nx = int(np.ceil(2.*np.sqrt(self.num_modes / 2.)))
            ny = int(np.ceil(self.num_modes / float(nx)))
            total_nx = nx + int(np.ceil(nx / 2)) + 1
            gs = matplotlib.gridspec.GridSpec(ny, total_nx)
            gs.update(wspace=0.02, hspace=0.02)
            self.subplot_list = []
            for m in range(self.num_modes):
                s = self.fig.add_subplot(gs[m/nx, m%nx])
                s.imshow(self.vol[m]**exponent, vmin=rangemin, vmax=rangemax, cmap=cmap, interpolation='none')
                #s.set_title('Class %d'%m)
                s.text(0.05, 0.85, '%d'%m, transform=s.transAxes, fontsize=10, color='w', bbox={'facecolor': 'black', 'pad': 0})
                s.axis('off')
                self.subplot_list.append(s)
            s = self.fig.add_subplot(gs[:,nx:])
            s.imshow(self.vol[num]**exponent, vmin=rangemin, vmax=rangemax, cmap=cmap, interpolation='none')
            s.set_title('Class %d'%num)
            s.axis('off')
            self.fig.canvas.mpl_connect('button_press_event', self.select_mode)

        self.canvas.draw()
        self.image_exists = True
        self.need_replot = False

    def parse(self):
        fname = self.fname.text()

        if os.path.isfile(fname):
            f = open(fname, "r")
        else:
            sys.stderr.write("Unable to open %s\n"%fname)
            return

        self.vol = np.fromfile(f, dtype='f8')
        if self.num_modes == 0:
            self.size = int(np.ceil(np.power(len(self.vol), 1./3.)))
            self.vol = self.vol.reshape(self.size, self.size, self.size)
            self.center = self.size/2
            if not self.image_exists:
                self.layer_slider.setRange(0, self.size-1)
                self.layernum.setMaximum(self.size-1)
                self.layer_slider.setValue(self.center)
                self.layerslider_moved(self.center)
        else:
            self.size = int(np.ceil(np.power(len(self.vol)/self.num_modes, 1./2.)))
            self.vol = self.vol.reshape(self.num_modes, self.size, self.size)
            self.center = 0
            if not self.image_exists:
                self.layer_slider.setRange(0, self.num_modes-1)
                self.layernum.setMaximum(self.num_modes-1)
                self.layer_slider.setValue(self.center)
                self.layerslider_moved(self.center)

        self.old_fname = fname

    def plot_log(self):
        # Read log file to get log lines (one for each completed iteration)
        with open(self.logfname.text(), 'r') as f:
            all_lines = f.readlines()
            self.emclog_text.setText(''.join(all_lines))

            lines = [l.rstrip().split() for l in all_lines]
            flag = False
            loglines = []
            for l in lines:
                if len(l) < 1:
                    continue
                if flag is True:
                    loglines.append(l)
                elif l[0].isdigit():
                    flag = True
                    loglines.append(l)

        loglines = np.array(loglines)
        if len(loglines) == 0:
            return

        # Read orientation files for the first n iterations
        o_files = sorted(glob.glob(self.folder+"/orientations/*.bin"))
        self.orient = []
        for i in range(len(loglines)):
            p = self.folder+'/orientations/orientations_%.3d.bin' % (i+1)
            fn = os.path.split(p)[-1]
            with open(p, 'r') as f:
                self.orient.append(np.fromfile(f, '=i4'))
        olengths = np.array([len(o) for o in self.orient])
        max_length = olengths.max()

        iternum = loglines[:,0].astype(np.int32)
        change = loglines[:,2].astype(np.float64)
        info = loglines[:,3].astype(np.float64)
        like = loglines[:,4].astype(np.float64)
        num_rot = loglines[:,5].astype(np.int32)
        beta = loglines[:,6].astype(np.float64)
        num_rot_change = np.append(np.where(np.diff(num_rot)>0)[0], num_rot.shape[0])
        beta_change = np.where(np.diff(beta)>0.)[0]

        # Sort o_array by the last iteration which has the same number of orientations
        o_array = np.array([np.pad(o, ((max_length-len(o),0)), 'constant', constant_values=-1) for o in self.orient]).astype('f8')
        #o_array = np.asarray(self.orient, dtype='f8')
        istart = 0
        for i in range(len(num_rot_change)):
            istop = num_rot_change[i]
            ord = o_array[istop-1].argsort()
            for index in np.arange(istart,istop):
                o_array[index] = o_array[index][ord]
                #o_array[index] /= float(num_rot[index])
            istart = istop
        o_array = o_array.T

        self.log_fig.clf()
        grid = matplotlib.gridspec.GridSpec(2,3, wspace=0.3, hspace=0.2)
        grid.update(left=0.05, right=0.99, hspace=0.2, wspace=0.3)

        # Plot RMS change
        s1 = self.log_fig.add_subplot(grid[:,0])
        s1.plot(iternum, change, 'o-')
        s1.set_yscale('log')
        s1.set_xlabel('Iteration')
        s1.set_ylabel('RMS change', labelpad=-10)
        s1_lim = s1.get_ylim()
        s1.set_ylim(s1_lim)
        for i in beta_change:
            s1.plot([i+1,i+1], s1_lim,'k--',lw=1)
        for i in num_rot_change[:-1]:
            s1.plot([i+1,i+1], s1_lim,'r--',lw=1)

        # Plot average mutual information
        s2 = self.log_fig.add_subplot(grid[0,1])
        s2.plot(iternum, info, 'o-')
        s2.set_xlabel('Iteration')
        s2.set_ylabel(r'Mutual info. $I(K,\Omega | W)$')
        s2_lim = s2.get_ylim()
        s2.set_ylim(s2_lim)
        for i in beta_change:
            s2.plot([i+1,i+1], s2_lim,'k--',lw=1)
        for i in num_rot_change[:-1]:
            s2.plot([i+1,i+1], s2_lim,'r--',lw=1)

        # Plot average log-likelihood
        s3 = self.log_fig.add_subplot(grid[1,1])
        s3.plot(iternum, like, 'o-')
        s3.set_xlabel('Iteration')
        s3.set_ylabel('Avg log-likelihood')
        s3_lim = s3.get_ylim()
        s3.set_ylim(s3_lim)
        for i in beta_change:
            s3.plot([i+1,i+1], s3_lim,'k--',lw=1)
        for i in num_rot_change[:-1]:
            s3.plot([i+1,i+1], s3_lim,'r--',lw=1)

        # Plot most likely orientation convergence plot
        if len(loglines) > 1:
            s4 = self.log_fig.add_subplot(grid[:,2])
            o_array = o_array[o_array[:,-1]>=0]
            sh = o_array.shape
            s4.imshow(o_array**0.5, aspect=(1.*sh[1]/sh[0]), extent=[1,sh[1],sh[0],0], cmap=self.color_map.checkedAction().text())
            s4.get_yaxis().set_ticks([])
            s4.set_xlabel('Iteration')
            s4.set_ylabel('Pattern number (sorted)')
            s4.set_title('Most likely orientations of data\n(sorted/colored by last iteration)')

        grid.tight_layout(self.log_fig)
        self.plotcanvas.draw()

    def parse_and_plot(self, event=None):
        if not self.image_exists or self.old_fname != self.fname.text():
            self.parse()
            self.plot_vol(int(self.layernum.text()))
        elif self.need_replot:
            self.plot_vol(int(self.layernum.text()))
        else:
            pass

    def check_for_new(self, event=None):
        with open(self.logfname.text(), 'r') as f:
            last_line = f.readlines()[-1].rstrip().split()
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
            self.iterslider_moved(iteration)
            self.plot_log()
            self.parse_and_plot()

    def keep_checking(self, event=None):
        if self.ifcheck.isChecked():
            self.check_for_new()
            self.checker = QtCore.QTimer(self)
            self.checker.timeout.connect(self.check_for_new)
            self.checker.start(5000)
        else:
            self.checker.stop()

    def select_mode(self, event=None):
        curr_mode = -1
        for i, ax in enumerate(self.subplot_list):
            if event.inaxes is ax:
                curr_mode = i
        if curr_mode >= 0 and curr_mode != self.layernum.value():
            self.layer_slider.setValue(curr_mode)
            self.layernum.setValue(curr_mode)
            self.plot_vol(curr_mode)

    def force_plot(self, event=None):
        self.parse()
        self.plot_vol(int(self.layernum.text()))

    def load_volume(self, event=None):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load 3D Volume', 'data/', 'Binary data (*.bin)')
        if fname:
            self.fname.setText(fname)
            self.parse_and_plot()
        '''
        dialog = QtWidgets.QFileDialog(self)
        dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        #dialog.setFilter('3D Binary Volumes (*.bin)')
        if dialog._exec():
            self.fname.setText(dialog.selectedFiles())
            self.parse_and_plot()
        '''

    def save_plot(self, event=None):
        default_name = 'images/'+os.path.splitext(os.path.basename(self.fname.text()))[0]+'.png'
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Volume Image', default_name, 'Image (*.png)')
        if fname:
            self.fig.savefig(fname, bbox_inches='tight')
            sys.stderr.write('Saved to %s\n'%fname)

    def save_log_plot(self, event=None):
        default_name = 'images/log_fig.png'
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Log Plots', default_name, 'Image (*.png)')
        if fname:
            self.log_fig.savefig(fname, bbox_inches='tight')
            sys.stderr.write("Saved to %s\n"%fname)

    def cmap_changed(self, event=None):
        self.need_replot = True
        self.parse_and_plot()

    def keyPressEvent(self, event):
        k = event.key()
        m = int(event.modifiers())
        
        if k == QtCore.Qt.Key_Return or k == QtCore.Qt.Key_Enter:
            self.parse_and_plot()
        elif QtGui.QKeySequence(m+k) == QtGui.QKeySequence('Ctrl+Q'):
            self.close()
        elif QtGui.QKeySequence(m+k) == QtGui.QKeySequence('Ctrl+S'):
            self.save_plot()
        else:
            event.ignore()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dragonfly Progress Monitor')
    parser.add_argument('-c', '--config_file', help='Path to config file. Default=config.ini', default='config.ini')
    parser.add_argument('-f', '--volume_file', help='Show slices of particular file instead of output', default=None)
    args, unknown = parser.parse_known_args()
    
    app = QtWidgets.QApplication(unknown)
    app.setStyleSheet(qdarkstyle.load_stylesheet_from_environment())
    p = Progress_viewer(config=args.config_file, model=args.volume_file)
    sys.exit(app.exec_())
