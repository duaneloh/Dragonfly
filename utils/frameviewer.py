#!/usr/bin/env python

import sys
import os
try:
    from PyQt5 import QtCore, QtWidgets, QtGui
    os.environ['QT_API'] = 'pyqt5'
except ImportError:
    import sip
    sip.setapi('QString', 2)
    from PyQt4 import QtCore, QtGui
    from PyQt4 import QtGui as QtWidgets
    os.environ['QT_API'] = 'pyqt'
import qdarkstyle
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
            self.cmap = 'coolwarm'
        else:
            self.cmap = cmap
        self.mode_val = None
        
        read_config.read_gui_config(self, 'emc')
        py_utils.gen_det_and_emc(self)
        self.init_UI()

    def init_UI(self):
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('Dragonfly Frame Viewer')
        window = QtWidgets.QWidget()

        self.hbox = QtWidgets.QHBoxLayout()
        self.frame_panel = frame_panel.FramePanel(self, powder=self.do_powder, compare=self.do_compare)
        self.hbox.addWidget(self.frame_panel)

        # Menu items
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        # Color map picker
        cmapmenu = menubar.addMenu('&Color Map')
        self.color_map = QtWidgets.QActionGroup(self, exclusive=True)
        for i, s in enumerate(['coolwarm', 'cubehelix', 'CMRmap', 'gray', 'gray_r', 'jet']):
            a = self.color_map.addAction(QtWidgets.QAction(s, self, checkable=True))
            if i == 0:
                a.setChecked(True)
            a.triggered.connect(self.cmap_changed)
            cmapmenu.addAction(a)

        window.setLayout(self.hbox)
        self.setCentralWidget(window)
        self.show()

    def cmap_changed(self, event=None):
        self.cmap = self.color_map.checkedAction().text()
        self.frame_panel.plot_frame()

    def keyPressEvent(self, event):
        k = event.key()
        m = int(event.modifiers())
        
        if QtGui.QKeySequence(m+k) == QtGui.QKeySequence('Ctrl+Q'):
            self.close()
        else:
            event.ignore()

if __name__ == '__main__':
    parser = py_utils.MyArgparser(description='Utility for viewing frames of the emc file (list)')
    parser.add_argument('--cmap', help='Matplotlib color map (default: coolwarm)')
    parser.add_argument('-M', '--mask', help='Whether to zero out masked pixels (default False)', action='store_true', default=False)
    parser.add_argument('-P', '--powder', help='Show powder sum of all frames', action='store_true', default=False)
    parser.add_argument('-C', '--compare', help='Compare with predicted intensities (needs data/quat.dat)', action='store_true', default=False)
    args = parser.special_parse_args()
    
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_from_environment())
    Frameviewer(args.config_file, cmap=args.cmap, mask=args.mask, do_powder=args.powder, do_compare=args.compare)
    sys.exit(app.exec_())
