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
    def __init__(self, config_file, cmap='coolwarm', mask=False, class_fname='my_classes.dat'):
        super(Classifier, self).__init__()
        if cmap is None:
            self.cmap = 'coolwarm'
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
        py_utils.gen_det_and_emc(self, classifier=True, mask=mask)
        self.classes = classes.FrameClasses(self.emc_reader.num_frames, fname=class_fname)
        
        self.init_UI()

    def init_UI(self):
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('Dragonfly Classifier')
        self.setGeometry(0, 0, 1100, 900)
        window = QtWidgets.QWidget()
        hbox = QtWidgets.QHBoxLayout()
        hbox.setSpacing(0)

        self.frame_panel = frame_panel.FramePanel(self)
        hbox.addWidget(self.frame_panel)

        window.setLayout(hbox)
        self.setCentralWidget(window)
        self.show()

        self.manual_panel = manual.ManualPanel(self)
        self.conversion_panel = conversion.ConversionPanel(self)
        self.embedding_panel = embedding.EmbeddingPanel(self)
        self.mlp_panel = mlp.MLPPanel(self)

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
        section = 'classifier'
        try:
            read_config.get_filename(self.config_file, section, 'nonexistent_option')
        except read_config.ConfigParser.NoSectionError:
            print 'No section named \'classifier\'. Taking parameters from \'emc\' section instead'
            section = 'emc'
        except read_config.ConfigParser.NoOptionError:
            pass
        
        read_config.read_gui_config(self, section)

    def tab_changed(self, index):
        self.mode_val = index
        self.frame_panel.plot_frame()

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
    parser = py_utils.MyArgparser(description='Data classifier')
    parser.add_argument('--cmap', help='Matplotlib color map (default: CMRmap)')
    parser.add_argument('-M', '--mask', help='Whether to zero out masked pixels (default False)', action='store_true', default=False)
    parser.add_argument('-C', '--class_fname', help='File containing classes for each frame', default='my_classes.dat')
    args = parser.special_parse_args()
    
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_from_environment())
    Classifier(args.config_file, cmap=args.cmap, mask=args.mask, class_fname=args.class_fname)
    sys.exit(app.exec_())
