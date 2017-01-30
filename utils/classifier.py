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
    def __init__(self, config_file, cmap='CMRmap', mask=False):
        super(Classifier, self).__init__()
        if cmap is None:
            self.cmap = 'CMRmap'
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
        self.ang_corr = None
        self.mode_val = 0
        
        self.get_config_params()
        self.geom = readdet.Det_reader(self.det_fname, self.detd, self.ewald_rad, mask_flag=mask)
        self.emc_reader = reademc.EMC_reader(self.photons_list, self.geom.x, self.geom.y, self.geom.raw_mask)
        self.num_frames = self.emc_reader.num_frames
        self.classes = classes.Frame_classes(self.num_frames)
        
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

        menubar = self.menuBar()
        thememenu = menubar.addMenu('&Theme')
        self.theme = QtWidgets.QActionGroup(self, exclusive=True)
        for i, s in enumerate(map(str, list(QtWidgets.QStyleFactory.keys()))):
            a = self.theme.addAction(QtWidgets.QAction(s, self, checkable=True))
            if i == 0:
                a.setChecked(True)
            a.triggered.connect(self.switch_theme)
            thememenu.addAction(a)

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
        
        pm = read_config.get_detector_config(self.config_file)
        
        self.num_files = len(self.photons_list)
        self.frame_shape = (pm['dets_x'], pm['dets_y'])
        self.det_fname = read_config.get_filename(self.config_file, 'classifier', 'in_detector_file')
        output_folder = read_config.get_filename(self.config_file, 'classifier', 'output_folder')
        self.output_folder = os.path.realpath(output_folder)
        self.ewald_rad = pm['ewald_rad']
        self.detd = pm['detd']/pm['pixsize']
        self.blacklist = None

    def tab_changed(self, index):
        self.mode_val = index
        self.frame_panel.plot_frame()

    def switch_theme(self, event=None):
        QtWidgets.QApplication.instance().setStyle(self.theme.checkedAction().text())

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
                self.manual_panel.save_class_list()
                event.accept()
            elif result == QtWidgets.QMessageBox.No:
                event.accept()
            else:
                event.ignore()

if __name__ == '__main__':
    parser = py_utils.my_argparser(description='Utility for viewing frames of the emc file (list)')
    parser.add_argument('--cmap', help='Matplotlib color map (default: CMRmap)')
    parser.add_argument('-M', '--mask', help='Whether to zero out masked pixels (default False)', action='store_true', default=False)
    args = parser.special_parse_args()
    
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    Classifier(args.config_file, cmap=args.cmap, mask=args.mask)
    sys.exit(app.exec_())
