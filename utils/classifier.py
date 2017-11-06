#!/usr/bin/env python

import sys
import os
import numpy as np
try:
    from PyQt5 import QtCore, QtWidgets, QtGui
except ImportError:
    import sip
    sip.setapi('QString', 2)
    from PyQt4 import QtCore, QtGui
    from PyQt4 import QtGui as QtWidgets
from py_src import reademc
from py_src import readdet
from py_src import manual
from py_src import conversion
from py_src import embedding
from py_src import classes
from py_src import mlp
from py_src import py_utils
from py_src import read_config
from py_src import frame_panel

class Classifier(QtWidgets.QMainWindow):
    def __init__(self, config_file, cmap='CMRmap', mask=False, class_fname='my_classes.dat'):
        super(Classifier, self).__init__()
        if cmap is None:
            self.cmap = 'cubehelix'
        else:
            self.cmap = cmap
        self.config_file = config_file
        self.mode_dict = {
            '&Display': 0,
            '&Manual': 1,
            '&Convert': 2,
            '&Embedding': 3,
            'M&LP': 4
        }
        self.converted = None
        self.mode_val = 0
        
        self.get_config_params()
        if len(set(self.det_list)) == 1:
            geom_list = [readdet.Det_reader(self.det_list[0], self.detd, self.ewald_rad, mask_flag=mask)]
            geom_mapping = None
        else:
            print('The Classifier GUI will likely have problems with multiple geometries')
            print('We recommend classifying patterns with a common geometry')
            uniq = sorted(set(self.det_list))
            geom_list = [readdet.Det_reader(fname, self.detd, self.ewald_rad, mask_flag=mask) for fname in uniq]
            geom_mapping = [uniq.index(fname) for fname in self.det_list]
        self.geom = geom_list[0]
        self.emc_reader = reademc.EMC_reader(self.photons_list, geom_list, geom_mapping) 
        self.num_frames = self.emc_reader.num_frames
        self.classes = classes.Frame_classes(self.num_frames, fname=class_fname)
        
        self.init_UI()

    def init_UI(self):
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('Dragonfly Classifier')
        self.setGeometry(0,0,1100,900)
        window = QtWidgets.QWidget()
        hbox = QtWidgets.QHBoxLayout()
        hbox.setSpacing(0)

        self.frame_panel = frame_panel.Frame_panel(self)
        hbox.addWidget(self.frame_panel)

        window.setLayout(hbox)
        self.setCentralWidget(window)
        self.show()

        self.manual_panel = manual.Manual_panel(self)
        self.conversion_panel = conversion.Conversion_panel(self)
        self.embedding_panel = embedding.Embedding_panel(self)
        self.mlp_panel = mlp.MLP_panel(self)

        # Menu items
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        # Theme picker
        thememenu = menubar.addMenu('&Theme')
        self.theme = QtWidgets.QActionGroup(self, exclusive=True)
        for i, s in enumerate(map(str, list(QtWidgets.QStyleFactory.keys()))):
            a = self.theme.addAction(QtWidgets.QAction(s, self, checkable=True))
            if i == 0:
                a.setChecked(True)
            a.triggered.connect(self.theme_changed)
            thememenu.addAction(a)
        # Color map picker
        cmapmenu = menubar.addMenu('&Color Map')
        self.color_map = QtWidgets.QActionGroup(self, exclusive=True)
        for i, s in enumerate(['cubehelix', 'CMRmap', 'gray', 'gray_r', 'jet']):
            a = self.color_map.addAction(QtWidgets.QAction(s, self, checkable=True))
            if i == 0:
                a.setChecked(True)
            a.triggered.connect(self.cmap_changed)
            cmapmenu.addAction(a)

        toolbox = QtWidgets.QToolBox(self)
        hbox.addWidget(toolbox)
        toolbox.setFixedWidth(300)
        toolbox.addItem(QtWidgets.QWidget(self), '&Display')
        toolbox.addItem(self.manual_panel, '&Manual')
        toolbox.addItem(self.conversion_panel, '&Conversion')
        toolbox.addItem(self.embedding_panel, '&Embedding')
        toolbox.addItem(self.mlp_panel, 'M&LP')
        toolbox.currentChanged.connect(self.tab_changed)

    def get_config_params(self):
        try:
            pfile = read_config.get_filename(self.config_file, 'classifier', 'in_photons_file')
            print 'Using in_photons_file: %s' % pfile
            self.photons_list = [pfile]
        except read_config.ConfigParser.NoOptionError:
            plist = read_config.get_filename(self.config_file, 'classifier', 'in_photons_list')
            print 'Using in_photons_list: %s' % plist
            with open(plist, 'r') as f:
                self.photons_list = map(lambda x: x.rstrip(), f.readlines())
                self.photons_list = [line for line in self.photons_list if line]
        try:
            dfile = read_config.get_filename(self.config_file, 'classifier', 'in_detector_file')
            print 'Using in_detector_file: %s' % dfile
            self.det_list = [dfile]
        except read_config.ConfigParser.NoOptionError:
            dlist = read_config.get_filename(self.config_file, 'classifier', 'in_detector_list')
            print 'Using in_detector_list: %s' % dlist
            with open(dlist, 'r') as f:
                self.det_list = map(lambda x: x.rstrip(), f.readlines())
                self.det_list = [line for line in self.det_list if line]
        if len(self.det_list) > 1 and len(self.det_list) != len(self.photons_list):
            raise ValueError('Different number of detector and photon files')
        
        # Only used with old detector file
        pm = read_config.get_detector_config(self.config_file)
        self.ewald_rad = pm['ewald_rad']
        self.detd = pm['detd']/pm['pixsize']
        
        self.num_files = len(self.photons_list)
        output_folder = read_config.get_filename(self.config_file, 'classifier', 'output_folder')
        self.output_folder = os.path.abspath(output_folder)
        self.blacklist = None

    def tab_changed(self, index):
        self.mode_val = index
        self.frame_panel.plot_frame()

    def theme_changed(self, event=None):
        QtWidgets.QApplication.instance().setStyle(self.theme.checkedAction().text())

    def cmap_changed(self, event=None):
        self.cmap = self.color_map.checkedAction().text()
        self.frame_panel.plot_frame()

    def keyPressEvent(self, event):
        k = event.key()
        m = int(event.modifiers())
        
        if QtGui.QKeySequence(m+k) == QtGui.QKeySequence('Ctrl+N'):
            self.frame_panel.next_frame()
        elif QtGui.QKeySequence(m+k) == QtGui.QKeySequence('Ctrl+P'):
            self.frame_panel.prev_frame()
        elif QtGui.QKeySequence(m+k) == QtGui.QKeySequence('Ctrl+R'):
            self.frame_panel.rand_frame()
        elif QtGui.QKeySequence(m+k) == QtGui.QKeySequence('Ctrl+Q'):
            self.close()
        else:
            event.ignore()

    def closeEvent(self, event):
        self.quit(event)

    def quit(self, event):
        if self.classes.unsaved:
            result = QtWidgets.QMessageBox.question(
                self, 
                'Warning', 
                'Unsaved changes to class list. Save?',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel,
                QtWidgets.QMessageBox.Cancel)
            if result == QtWidgets.QMessageBox.Yes:
                self.classes.save()
                event.accept()
            elif result == QtWidgets.QMessageBox.No:
                event.accept()
            else:
                event.ignore()

if __name__ == '__main__':
    parser = py_utils.my_argparser(description='Data classifier')
    parser.add_argument('--cmap', help='Matplotlib color map (default: CMRmap)')
    parser.add_argument('-M', '--mask', help='Whether to zero out masked pixels (default False)', action='store_true', default=False)
    parser.add_argument('-C', '--class_fname', help='File containing classes for each frame', default='my_classes.dat')
    args = parser.special_parse_args()
    
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    Classifier(args.config_file, cmap=args.cmap, mask=args.mask, class_fname=args.class_fname)
    sys.exit(app.exec_())
