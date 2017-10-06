#!/usr/bin/env python

import numpy as np
import sys
import os
try:
    from PyQt5 import QtCore, QtWidgets, QtGui
except ImportError:
    import sip
    sip.setapi('QString', 2)
    from PyQt4 import QtCore, QtGui
    from PyQt4 import QtGui as QtWidgets
from py_src import py_utils
from py_src import read_config
from py_src import frame_panel
from py_src import reademc 
from py_src import readdet

class Frameviewer(QtWidgets.QMainWindow):
    def __init__(self, config_file, cmap=None, mask=False, do_powder=False, do_compare=False):
        super(Frameviewer, self).__init__()
        self.config_file = config_file
        self.do_powder = do_powder
        self.do_compare = do_compare
        if cmap is None:
            self.cmap = 'CMRmap'
        else:
            self.cmap = cmap
        self.mode_val = None
        
        self.get_config_params()
        if len(set(self.det_list)) == 1:
            geom_list = [readdet.Det_reader(self.det_list[0], self.detd, self.ewald_rad, mask_flag=mask)]
            geom_mapping = None
        else:
            uniq = sorted(set(self.det_list))
            geom_list = [readdet.Det_reader(fname, self.detd, self.ewald_rad, mask_flag=mask) for fname in uniq]
            geom_mapping = [uniq.index(fname) for fname in self.det_list]
        self.geom = geom_list[0]
        self.emc_reader = reademc.EMC_reader(self.photons_list, geom_list, geom_mapping) 
        self.num_frames = self.emc_reader.num_frames
        self.init_UI()

    def init_UI(self):
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('Dragonfly Frame Viewer')
        window = QtWidgets.QWidget()
        self.hbox = QtWidgets.QHBoxLayout()
        
        self.frame_panel = frame_panel.Frame_panel(self, powder=self.do_powder, compare=self.do_compare)
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
        self.num_files = len(self.photons_list)
        try:
            dfile = read_config.get_filename(self.config_file, 'emc', 'in_detector_file')
            print 'Using in_detector_file: %s' % dfile
            self.det_list = [dfile]
        except read_config.ConfigParser.NoOptionError:
            dlist = read_config.get_filename(self.config_file, 'emc', 'in_detector_list')
            print 'Using in_detector_list: %s' % dlist
            with open(dlist, 'r') as f:
                self.det_list = map(lambda x: x.rstrip(), f.readlines())
        if len(self.det_list) > 1 and len(self.det_list) != len(self.photons_list):
            raise ValueError('Different number of detector and photon files')
        
        # Only used with old detector file
        pm = read_config.get_detector_config(self.config_file)
        self.ewald_rad = pm['ewald_rad']
        self.detd = pm['detd']/pm['pixsize']
        
        self.log_fname = read_config.get_filename(self.config_file, 'emc', 'log_file')
        try:
            output_folder = read_config.get_filename(self.config_file, 'emc', 'output_folder')
        except read_config.ConfigParser.NoOptionError:
            output_folder = 'data/'
        self.output_folder = os.path.realpath(output_folder)
        
        try:
            self.blacklist = np.loadtxt(read_config.get_filename(args.config_file, 'emc', 'blacklist_file'), dtype='u1')
        except read_config.ConfigParser.NoOptionError:
            self.blacklist = None

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
    parser.add_argument('--cmap', help='Matplotlib color map (default: CMRmap)')
    parser.add_argument('-M', '--mask', help='Whether to zero out masked pixels (default False)', action='store_true', default=False)
    parser.add_argument('-P', '--powder', help='Show powder sum of all frames', action='store_true', default=False)
    parser.add_argument('-C', '--compare', help='Compare with predicted intensities (needs data/quat.dat)', action='store_true', default=False)
    args = parser.special_parse_args()
    
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    Frameviewer(args.config_file, cmap=args.cmap, mask=args.mask, do_powder=args.powder, do_compare=args.compare)
    sys.exit(app.exec_())
