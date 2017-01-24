#!/usr/bin/env python

import numpy as np
import sys
import os
try:
    from PyQt5 import QtCore, QtWidgets, QtGui
except ImportError:
    from PyQt4 import QtCore, QtGui
    from PyQt4 import QtGui as QtWidgets
from py_src import py_utils
from py_src import read_config
from py_src import frame_panel
from py_src import reademc 
from py_src import readdet

class Frameviewer(QtWidgets.QMainWindow):
    def __init__(self, config_file, cmap='jet', mask=False):
        super(Frameviewer, self).__init__()
        self.config_file = config_file
        self.cmap = cmap
        self.mode_val = None
        
        self.get_config_params()
        self.geom = readdet.Det_reader(self.det_fname, self.detd, self.ewald_rad, mask_flag=mask)
        self.emc_reader = reademc.EMC_reader(self.photons_list, self.geom.x, self.geom.y, self.geom.mask)
        self.num_frames = self.emc_reader.num_frames
        self.init_UI()

    def init_UI(self):
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('Dragonfly Frame Viewer')
        window = QtWidgets.QWidget()
        self.hbox = QtWidgets.QHBoxLayout()
        
        self.frame_panel = frame_panel.Frame_panel(self)
        self.hbox.addWidget(self.frame_panel)
        
        window.setLayout(self.hbox)
        self.setCentralWidget(window)
        self.show()

    def get_config_params(self):
        try:
            pfile = read_config.get_filename(self.config_file, 'emc', 'in_photons_file')
            print 'Using in_photons_file: %s' % pfile
            self.photons_list = [pfile]
        except read_config.ConfigParser.NoOptionError:
            plist = read_config.get_filename(self.config_file, 'emc', 'in_photons_list')
            print 'Using in_photons_list: %s' % plist
            with open(plist, 'r') as f:
                self.photons_list = map(lambda x: x.rstrip(), f.readlines())
        
        pm = read_config.get_detector_config(self.config_file)
        
        self.num_files = len(self.photons_list)
        self.frame_shape = (pm['dets_x'], pm['dets_y'])
        self.det_fname = read_config.get_filename(self.config_file, 'emc', 'in_detector_file')
        output_folder = read_config.get_filename(self.config_file, 'emc', 'output_folder')
        try:
            self.blacklist = np.loadtxt(read_config.get_filename(args.config_file, 'emc', 'blacklist_file'), dtype='u1')
        except read_config.ConfigParser.NoOptionError:
            self.blacklist = None
        
        self.output_folder = os.path.realpath(output_folder)
        self.ewald_rad = pm['ewald_rad']
        self.detd = pm['detd']/pm['pixsize']

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

if __name__ == '__main__':
    parser = py_utils.my_argparser(description='Utility for viewing frames of the emc file (list)')
    parser.add_argument('--cmap', help='Matplotlib color map (default: jet)')
    parser.add_argument('-M', '--mask', help='Whether to zero out masked pixels (default False)', action='store_true', default=False)
    args = parser.special_parse_args()
    
    app = QtWidgets.QApplication(sys.argv)
    print QtWidgets.QStyleFactory.keys()
    app.setStyle('Fusion')
    Frameviewer(args.config_file, cmap=args.cmap, mask=args.mask)
    sys.exit(app.exec_())
