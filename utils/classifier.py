#!/usr/bin/env python

'''Module containing Classifier GUI'''

from __future__ import print_function
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
from py_src import manual
from py_src import conversion
from py_src import embedding
from py_src import classes
from py_src import mlp
from py_src import py_utils
from py_src import read_config
from py_src import frame_panel

class Classifier(QtWidgets.QMainWindow):
    '''Classifier GUI which can be used to classify frames before 3D merging
    The GUI is made up of many panels representing different methods, each of whom assigns one
    of the 26 lower case alphabets as the class of a frame.
    Techniques:
        Manual - Manually classify patterns using keyboard
        Conversion - Convert frames to different basis to ease classification
        Embedding - Manifold embedding of converted frames
        MLP - Multi-layer perceptron neural network classifier
    '''
    def __init__(self, config_file, mask=False):
        super(Classifier, self).__init__()
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
        self.cmap = None

        self._get_config_params()
        if self.stack_size == 0:
            py_utils.gen_det_and_emc(self, classifier=True, mask=mask)
        else:
            py_utils.gen_stack(self)
        self.classes = classes.FrameClasses(self.emc_reader.num_frames, fname=self.class_fname)

        self._init_ui()

    def _init_ui(self):
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('Dragonfly Classifier')
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'py_src/style.css'), 'r') as f:
            self.css = f.read()
        self.setStyleSheet(self.css)
        self.setGeometry(0, 0, 1100, 900)
        window = QtWidgets.QWidget()
        window.setObjectName('frame')
        hbox = QtWidgets.QHBoxLayout()
        hbox.setSpacing(0)
        hbox.setContentsMargins(0, 0, 0, 0)

        # Menu items
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        cmapmenu = menubar.addMenu('&Color Map')
        self.color_map = QtWidgets.QActionGroup(self, exclusive=True)
        cmaplist = ['coolwarm', 'cubehelix', 'CMRmap', 'gray', 'gray_r', 'jet']
        for i, cmap in enumerate(cmaplist):
            action = self.color_map.addAction(QtWidgets.QAction(cmap, self, checkable=True))
            if i == 0:
                action.setChecked(True)
            action.triggered.connect(self._cmap_changed)
            cmapmenu.addAction(action)
        self.cmap = cmaplist[0]

        self.frame_panel = frame_panel.FramePanel(self)
        hbox.addWidget(self.frame_panel)

        self.manual_panel = manual.ManualPanel(self)
        self.conversion_panel = conversion.ConversionPanel(self)
        self.embedding_panel = embedding.EmbeddingPanel(self)
        self.mlp_panel = mlp.MLPPanel(self)

        toolbox = QtWidgets.QToolBox(self)
        hbox.addWidget(toolbox)
        toolbox.setFixedWidth(300)
        toolbox.addItem(QtWidgets.QWidget(self), '&Display')
        toolbox.addItem(self.manual_panel, '&Manual')
        toolbox.addItem(self.conversion_panel, '&Conversion')
        toolbox.addItem(self.embedding_panel, '&Embedding')
        toolbox.addItem(self.mlp_panel, 'M&LP')
        toolbox.currentChanged.connect(self._tab_changed)

        window.setLayout(hbox)
        self.setCentralWidget(window)
        self.show()

    def _get_config_params(self):
        section = 'classifier'
        try:
            read_config.get_filename(self.config_file, section, 'nonexistent_option')
        except read_config.configparser.NoSectionError:
            print('No section named \'classifier\'. Taking parameters from \'emc\' section instead')
            section = 'emc'
            self.class_fname = 'my_classes.dat'
            self.polar_params = ['5', '60', '2.', '10.']
        except read_config.configparser.NoOptionError:
            pass

        read_config.read_gui_config(self, section)

    def _tab_changed(self, index):
        self.mode_val = index
        self.frame_panel.plot_frame()

    def _cmap_changed(self):
        self.cmap = self.color_map.checkedAction().text()
        self.frame_panel.plot_frame()

    def keyPressEvent(self, event): # pylint: disable=C0103
        '''Override of default keyPress event handler'''
        key = event.key()
        mod = int(event.modifiers())

        if QtGui.QKeySequence(mod+key) == QtGui.QKeySequence('Ctrl+Q'):
            self.close()
        else:
            event.ignore()

    def closeEvent(self, event): # pylint: disable=C0103
        '''Override of default close event handler'''
        self._quit(event)

    def _quit(self, event):
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

def main():
    '''Parses command line arguments and launches Classifier GUI'''
    parser = py_utils.MyArgparser(description='Data classifier')
    parser.add_argument('-M', '--mask',
                        help='Whether to zero out masked pixels (default False)',
                        action='store_true', default=False)
    args = parser.special_parse_args()

    app = QtWidgets.QApplication(sys.argv)
    Classifier(args.config_file, mask=args.mask)
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
