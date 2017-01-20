#!/usr/bin/env python

import argparse
import numpy as np
import sys
import os
import time
import glob
import re
import sip
sip.setapi('QString', 2)
from PyQt4 import QtCore, QtGui
import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvas

class Progress_viewer(QtGui.QMainWindow):
    def __init__(self, config='config.ini', model=None):
        super(Progress_viewer, self).__init__()
        self.config = config
        self.model_name = model
        self.log_txt = ""
        self.max_iter = 0
        self.need_replot = False
        self.image_exists = False

        self.init_UI()
        self.read_config(config)
        if model is not None:
            self.parse_and_plot()
        self.old_fname = self.fname.text()

    def init_UI(self):
        self.setWindowTitle('Dragonfly Progress Viewer')
        overall = QtGui.QWidget()
        self.setCentralWidget(overall)
        self.grid = QtGui.QGridLayout(overall)

        # Volume slices figure
        self.fig = matplotlib.figure.Figure(figsize=(15,5), facecolor='w')
        self.fig.subplots_adjust(left=0.0, bottom=0.00, right=0.99, wspace=0.0)
        self.canvas = FigureCanvas(self.fig)
        self.grid.addWidget(self.canvas, 0, 0)
        self.canvas.show()

        # Progress plots figure
        self.log_fig = matplotlib.figure.Figure(figsize=(15,5), facecolor='white')
        self.plotcanvas = FigureCanvas(self.log_fig)
        self.grid.addWidget(self.plotcanvas, 1, 0)
        self.plotcanvas.show()

        # Plot options widget
        self.options = QtGui.QVBoxLayout()
        self.grid.addLayout(self.options, 0, 1, 2, 1)

        hbox = QtGui.QHBoxLayout()
        self.options.addLayout(hbox)
        label = QtGui.QLabel('Log file name:', self)
        hbox.addWidget(label)
        self.logfname = QtGui.QLineEdit('EMC.log', self)
        self.logfname.setMinimumWidth(160)
        hbox.addWidget(self.logfname)
        label = QtGui.QLabel('PlotMax:', self)
        hbox.addWidget(label)
        self.rangestr = QtGui.QLineEdit('1', self)
        self.rangestr.setFixedWidth(48)
        self.rangestr.returnPressed.connect(self.range_changed)
        hbox.addWidget(self.rangestr)

        hbox = QtGui.QHBoxLayout()
        self.options.addLayout(hbox)
        label = QtGui.QLabel('File name:', self)
        hbox.addWidget(label)
        self.fname = QtGui.QLineEdit('data/output/intens_001.bin', self)
        self.logfname.setMinimumWidth(160)
        hbox.addWidget(self.fname)
        label = QtGui.QLabel('Exp:', self)
        hbox.addWidget(label)
        self.expstr = QtGui.QLineEdit('1', self)
        self.expstr.setFixedWidth(48)
        self.expstr.returnPressed.connect(self.range_changed)
        hbox.addWidget(self.expstr)

        hbox = QtGui.QHBoxLayout()
        self.options.addLayout(hbox)
        label = QtGui.QLabel('Image name:', self)
        hbox.addWidget(label)
        self.imagename = QtGui.QLineEdit('images/'+os.path.splitext(os.path.basename(self.fname.text()))[0]+'.png', self)
        self.imagename.setMinimumWidth(160)
        hbox.addWidget(self.imagename)
        button = QtGui.QPushButton('Save', self)
        button.clicked.connect(self.save_plot)
        hbox.addWidget(button)

        hbox = QtGui.QHBoxLayout()
        self.options.addLayout(hbox)
        label = QtGui.QLabel('Log image name:', self)
        hbox.addWidget(label)
        self.log_imagename = QtGui.QLineEdit('images/log_fig.png', self)
        self.log_imagename.setMinimumWidth(160)
        hbox.addWidget(self.log_imagename)
        button = QtGui.QPushButton('Save', self)
        button.clicked.connect(self.save_log_plot)
        hbox.addWidget(button)

        hbox = QtGui.QHBoxLayout()
        self.options.addLayout(hbox)
        label = QtGui.QLabel('Layer num.', self)
        hbox.addWidget(label)
        self.layer_slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.layer_slider.setRange(0, 200)
        self.layer_slider.sliderMoved.connect(self.layernum_changed)
        self.layer_slider.sliderReleased.connect(self.layernum_changed)
        hbox.addWidget(self.layer_slider)
        self.layernum = QtGui.QLineEdit(str(self.layer_slider.value()), self)
        self.layernum.returnPressed.connect(self.layernum_changed)
        self.layernum.setFixedWidth(36)
        hbox.addWidget(self.layernum)

        hbox = QtGui.QHBoxLayout()
        self.options.addLayout(hbox)
        label = QtGui.QLabel('Iteration', self)
        hbox.addWidget(label)
        self.iter_slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.iter_slider.setRange(0, 200)
        self.iter_slider.sliderMoved.connect(self.iter_changed)
        self.iter_slider.sliderReleased.connect(self.iter_changed)
        hbox.addWidget(self.iter_slider)
        self.iter = QtGui.QLineEdit(str(self.iter_slider.value()), self)
        self.iter.returnPressed.connect(self.iter_changed)
        self.iter.setFixedWidth(36)
        hbox.addWidget(self.iter)

        hbox = QtGui.QHBoxLayout()
        self.options.addLayout(hbox)
        button = QtGui.QPushButton('Check', self)
        button.clicked.connect(self.check_for_new)
        hbox.addWidget(button)
        self.ifcheck = QtGui.QCheckBox('Keep checking', self)
        self.ifcheck.stateChanged.connect(self.keep_checking)
        self.ifcheck.setChecked(False)
        hbox.addWidget(self.ifcheck)
        hbox.addStretch(1)

        hbox = QtGui.QHBoxLayout()
        self.options.addLayout(hbox)
        hbox.addStretch(1)
        button = QtGui.QPushButton('Plot', self)
        button.clicked.connect(self.parse_and_plot)
        hbox.addWidget(button)
        button = QtGui.QPushButton('Reparse', self)
        button.clicked.connect(self.force_plot)
        hbox.addWidget(button)
        button = QtGui.QPushButton('Quit', self)
        button.clicked.connect(self.close)
        hbox.addWidget(button)

        log_area = QtGui.QScrollArea(self)
        self.options.addWidget(log_area)
        log_area.setMinimumWidth(450)
        log_area.setWidgetResizable(True)
        self.emclog_text = QtGui.QTextEdit('Press Check to get log file contents', self)
        self.emclog_text.setReadOnly(True)
        self.emclog_text.setFontPointSize(8)
        self.emclog_text.setTabStopWidth(22)
        log_area.setWidget(self.emclog_text)

        self.show()

    def layernum_changed(self, value=None):
        if value is None:
            self.layer_slider.setValue(int(self.layernum.text()))
            self.need_replot = True
            self.parse_and_plot()
        else:
            self.layernum.setText(str(value))
            self.layer_slider.setValue(value)

    def iter_changed(self, value=None):
        if value is None:
            self.iter_slider.setValue(int(self.iter.text()))
            self.fname.setText('data/output/intens_%.3d.bin' % int(self.iter.text()))
            self.parse_and_plot()
        else:
            self.iter.setText('%3d'%value)
            self.iter_slider.setValue(value)
            self.fname.setText('data/output/intens_%.3d.bin' % value)

    def range_changed(self):
        self.need_replot = True

    def read_config(self, config):
        with open(config, 'r') as f:
            filestring = f.read()
            words = filter(None, re.split('[ =\n]', filestring))
            try:
                ind = words.index('output_folder')
                self.folder = words[ind+1]
            except ValueError:
                self.folder = 'data/'
            try:
                ind = words.index('log_file')
                self.logfname.setText(words[ind+1])
            except ValueError:
                self.logfname.setText('EMC.log')

    def plot_vol(self, num):
        self.imagename.setText('images/' + os.path.splitext(os.path.basename(self.fname.text()))[0] + '.png')
        rangemax = float(self.rangestr.text())
        exponent = float(self.expstr.text())

        a = self.vol[num,:,:]**exponent
        b = self.vol[:,num,:]**exponent
        c = self.vol[:,:,num]**exponent

        self.fig.clf()

        s1 = self.fig.add_subplot(131)
        s1.imshow(a, vmin=0, vmax=rangemax, cmap='CMRmap', interpolation='none')
        s1.set_title("YZ plane", y=1.01)

        s1.axis('off')

        s2 = self.fig.add_subplot(132)
        s2.matshow(b, vmin=0, vmax=rangemax, cmap='CMRmap', interpolation='none')
        s2.set_title("XZ plane", y=1.01)
        s2.axis('off')

        s3 = self.fig.add_subplot(133)
        s3.matshow(c, vmin=0, vmax=rangemax, cmap='CMRmap', interpolation='none')
        s3.set_title("XY plane", y=1.01)
        s3.axis('off')

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
        self.size = int(np.ceil(np.power(len(self.vol), 1./3.)))
        self.vol = self.vol.reshape(self.size, self.size, self.size)
        self.center = self.size/2
        if not self.image_exists:
            self.layer_slider.setRange(0, self.size-1)
            self.layernum_changed(self.center)

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

        iter = loglines[:,0].astype(np.int32)
        change = loglines[:,2].astype(np.float64)
        info = loglines[:,3].astype(np.float64)
        like = loglines[:,4].astype(np.float64)
        num_rot = loglines[:,5].astype(np.int32)
        beta = loglines[:,6].astype(np.float64)
        num_rot_change = np.append(np.where(np.diff(num_rot)>0)[0], num_rot.shape[0])
        beta_change = np.where(np.diff(beta)>0.)[0]

        # Sort o_array by the last iteration which has the same number of orientations
        o_array = np.asarray(self.orient, dtype='f8')
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
        grid.update(left=0.08, right=0.99, hspace=0.2, wspace=0.3)

        # Plot RMS change
        s1 = self.log_fig.add_subplot(grid[:,0])
        s1.plot(iter, change, 'o-')
        s1.set_yscale('log')
        s1.set_xlabel('Iteration')
        s1.set_ylabel('RMS change')
        s1_lim = s1.get_ylim()
        s1.set_ylim(s1_lim)
        for i in beta_change:
            s1.plot([i+1,i+1], s1_lim,'k--',lw=1)
        for i in num_rot_change[:-1]:
            s1.plot([i+1,i+1], s1_lim,'r--',lw=1)

        # Plot average mutual information
        s2 = self.log_fig.add_subplot(grid[0,1])
        s2.plot(iter, info, 'o-')
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
        s3.plot(iter[1:], like[1:], 'o-')
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
            sh = o_array.shape
            s4.imshow(o_array**0.5, aspect=(1.*sh[1]/sh[0]), extent=[1,sh[1],sh[0],0])
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

        if iteration > 0 and self.max_iter != iteration:
            self.fname.setText(self.folder+'/output/intens_%.3d.bin' % iteration)
            self.max_iter = iteration
            self.iter_slider.setRange(0, self.max_iter)
            self.iter_changed(iteration)
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

    def force_plot(self, event=None):
        self.parse()
        self.plot_vol(int(self.layernum.text()))

    def save_plot(self, event=None):
        self.fig.savefig(str(self.imagename.text()), bbox_inches='tight')
        sys.stderr.write('Saved to %s'%self.imagename.text())

    def save_log_plot(self, event=None):
        self.log_fig.savefig(self.log_imagename.text(), bbox_inches='tight')
        sys.stderr.write("Saved to %s\n"%self.log_imagename.text())

    def keyPressEvent(self, event):
        k = event.key()
        m = int(event.modifiers())
        
        if k == QtCore.Qt.Key_Return or k == QtCore.Qt.Key_Enter:
            self.parse_and_plot()
        elif QtGui.QKeySequence(m+k) == int(QtGui.QKeySequence('Ctrl+Q')):
            self.close()
        elif QtGui.QKeySequence(m+k) == int(QtGui.QKeySequence('Ctrl+S')):
            self.save_plot()
        else:
            event.ignore()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dragonfly Progress Monitor')
    parser.add_argument('-c', '--config_file', help='Path to config file. Default=config.ini', default='config.ini')
    parser.add_argument('-f', '--volume_file', help='Show slices of particular file instead of output', default=None)
    args = parser.parse_args()
    
    app = QtGui.QApplication(sys.argv)
    p = Progress_viewer(config=args.config_file, model=args.volume_file)
    sys.exit(app.exec_())
