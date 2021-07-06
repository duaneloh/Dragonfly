import os
try:
    from PyQt5 import QtCore, QtWidgets, QtGui # pylint: disable=import-error
    from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT #pylint: disable=no-name-in-module
    os.environ['QT_API'] = 'pyqt5'
except ImportError:
    import sip
    sip.setapi('QString', 2)
    from PyQt4 import QtCore, QtGui # pylint: disable=import-error
    from PyQt4 import QtGui as QtWidgets # pylint: disable=import-error
    from matplotlib.backends.backend_qt4agg import FigureCanvas, NavigationToolbar2QT #pylint: disable=no-name-in-module
    os.environ['QT_API'] = 'pyqt'

import numpy as np
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

from . import calc_cc

class CLPCA(QtWidgets.QMainWindow):
    windowClosed = QtCore.pyqtSignal()

    def __init__(self, output_fname, intens):
        super().__init__()
        self.output_fname = output_fname
        self.intens = intens
        self.calc = None
        self._init_ui()

    def _init_ui(self):
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('Common-line PCA')
        self.window = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout()
        self.window.setLayout(vbox)
        self.setCentralWidget(self.window)

        self.fig = Figure(figsize=(6, 6))
        self.fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.05)
        self.canvas = FigureCanvas(self.fig)
        vbox.addWidget(self.canvas)

        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Output file: %s'%self.output_fname, self)
        line.addWidget(label)
        label = QtWidgets.QLabel('(%d 2D averages)'%self.intens.shape[0], self)
        line.addWidget(label)
        line.addStretch(1)

        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('num_angbins:', self)
        line.addWidget(label)
        self.num_angbins = QtWidgets.QLineEdit('180', self)
        line.addWidget(self.num_angbins)
        label = QtWidgets.QLabel('min_radius:', self)
        line.addWidget(label)
        self.mask_radius = QtWidgets.QLineEdit('20', self)
        line.addWidget(self.mask_radius)
        button = QtWidgets.QPushButton('Calculate CC', self)
        button.clicked.connect(self._calculate_cc)
        line.addWidget(button)
        line.addStretch(1)

        self.show()

    def _calculate_cc(self):
        self.calc = calc_cc.CCCalculator(self.intens, 
                                         int(self.num_angbins.text()),
                                         int(self.mask_radius.text()),
                                         interp_order=0)
        best_cc = self.calc.run(nproc=1)
        try:
            self.calc.save_cc(self.output_fname)
        except OSError:
            print('Could not update', self.output_fname)

        pca = PCA(n_components=3)
        self.embedding = pca.fit_transform(best_cc)
        self._update_plot()

    def _update_plot(self):
        if self.calc is None:
            print('Calculate CC matrix first')
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.scatter(self.embedding[:,0], self.embedding[:,1], 
                   s=1, c=self.embedding[:,2])
        ax.set_facecolor('k')
        self.fig.tight_layout()
        self.canvas.draw()

    def closeEvent(self, event):
        self.windowClosed.emit()
        event.accept()

