#!/usr/bin/env python

'''Module containing Frameviewer class to view emc data frames'''
import sys
import os
try:
    from PyQt5 import QtCore, QtWidgets, QtGui # pylint: disable=import-error
    os.environ['QT_API'] = 'pyqt5'
except ImportError:
    import sip
    sip.setapi('QString', 2)
    from PyQt4 import QtCore, QtGui # pylint: disable=import-error
    from PyQt4 import QtGui as QtWidgets # pylint: disable=import-error
    os.environ['QT_API'] = 'pyqt'
from py_src import py_utils
from py_src import read_config
from py_src import frame_panel

class Frameviewer(QtWidgets.QMainWindow):
    '''GUI to view data frames.
    Optional modes:
        powder: Show sum of all frames
        compare: Compare against best tomogram
        mask: Apply mask to frames
    '''
    def __init__(self, config_file, mask=False, do_powder=False, do_compare=False, noscroll=False):
        super(Frameviewer, self).__init__()
        self.config_file = config_file
        self.do_powder = do_powder
        self.do_compare = do_compare
        self.noscroll = noscroll
        self.cmap = None

        read_config.read_gui_config(self, 'emc')
        py_utils.gen_det_and_emc(self, classifier=False, mask=mask)
        self._init_ui()

    def _init_ui(self):
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('Dragonfly Frame Viewer')
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'py_src/style.css'), 'r') as f:
            self.css = f.read()
        self.setStyleSheet(self.css)
        self.window = QtWidgets.QWidget()
        self.window.setObjectName('frame')

        # Menu bar
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        cmaplist = ['coolwarm', 'cubehelix', 'CMRmap', 'gray', 'gray_r', 'jet']
        cmapmenu = menubar.addMenu('&Color Map')
        self.color_map = QtWidgets.QActionGroup(self, exclusive=True)
        self.cmap = cmaplist[0]
        for i, cmap in enumerate(cmaplist):
            action = self.color_map.addAction(QtWidgets.QAction(cmap, self, checkable=True))
            if i == 0:
                action.setChecked(True)
            action.triggered.connect(self._cmap_changed)
            cmapmenu.addAction(action)

        # Frame panel
        self.vbox = QtWidgets.QVBoxLayout()
        self.vbox.setContentsMargins(0, 0, 0, 0)
        self.frame_panel = frame_panel.FramePanel(self, powder=self.do_powder,
                                                  compare=self.do_compare,
                                                  noscroll=self.noscroll)
        self.vbox.addWidget(self.frame_panel)

        self.window.setLayout(self.vbox)
        self.setCentralWidget(self.window)
        self.show()

    def _cmap_changed(self):
        self.cmap = self.color_map.checkedAction().text()
        '''
        if self.cmap == 'cubehelix':
            self.window.setObjectName('green')
            self.setStyleSheet(self.css)
        else:
            self.window.setObjectName('frame')
            self.setStyleSheet(self.css)
        '''
        self.frame_panel.plot_frame()

    def keyPressEvent(self, event): # pylint: disable=C0103
        '''Override of default keyPress event handler'''
        key = event.key()
        mod = int(event.modifiers())

        if QtGui.QKeySequence(mod+key) == QtGui.QKeySequence('Ctrl+Q'):
            self.close()
        else:
            event.ignore()

def main():
    '''Launch Frameviewer after parsing command line arguments'''
    parser = py_utils.MyArgparser(description='Utility for viewing frames of an emc file (list)')
    parser.add_argument('-M', '--mask',
                        help='Whether to zero out masked pixels (default False)',
                        action='store_true', default=False)
    parser.add_argument('-P', '--powder',
                        help='Show powder sum of all frames',
                        action='store_true', default=False)
    parser.add_argument('-C', '--compare',
                        help='Compare with predicted intensities (needs data/quat.dat)',
                        action='store_true', default=False)
    args = parser.special_parse_args()

    app = QtWidgets.QApplication(sys.argv)
    Frameviewer(args.config_file, mask=args.mask, do_powder=args.powder, do_compare=args.compare)
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
