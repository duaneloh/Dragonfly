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
import h5py
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

from . import calc_cc
from . import gui_utils

class CLPCA(QtWidgets.QMainWindow):
    windowClosed = QtCore.pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.output_fname = parent.fname.text()
        self.intens = parent.vol_plotter.vol

        self.cc_matrix = None
        self.embedding = None
        self._nearest_point = None

        self._init_ui()
        self._check_output()

    def _init_ui(self):
        if self.parent.css is not None:
            self.setStyleSheet(self.parent.css)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('Common-line PCA')
        self.window = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout()
        self.window.setLayout(vbox)
        self.setCentralWidget(self.window)
        self.window.setObjectName('frame')

        self.fig = Figure(figsize=(6, 6))
        self.fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.05)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.mpl_connect('button_release_event', self._get_nearest_class)
        self.navbar = gui_utils.MyNavigationToolbar(self.canvas, self)
        vbox.addWidget(self.navbar)
        vbox.addWidget(self.canvas, stretch=1)

        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Output file: %s'%self.output_fname, self)
        line.addWidget(label)
        label = QtWidgets.QLabel('(%d 2D averages)'%self.intens.shape[0], self)
        line.addWidget(label)
        line.addStretch(1)
        self.nearest_label = QtWidgets.QLabel('', self)
        line.addWidget(self.nearest_label)

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

    def _check_output(self):
        with h5py.File(self.output_fname, 'r') as fptr:
            if 'CL_CC' not in fptr:
                return
            self.cc_matrix = fptr['CL_CC/cc_matrix'][:]
            pca = PCA(n_components=3)
            self.embedding = pca.fit_transform(self.cc_matrix)

            self.num_angbins.setText(str(fptr['CL_CC/n_angbins'][...]))
            self.mask_radius.setText(str(fptr['CL_CC/mask_radius'][...]))

        self._update_plot()

    def _calculate_cc(self):
        self.calc = calc_cc.CCCalculator(self.intens,
                                         int(self.num_angbins.text()),
                                         int(self.mask_radius.text()),
                                         interp_order=0)
        self.cc_matrix = self.calc.run()
        try:
            self.calc.save_cc(self.output_fname)
        except OSError:
            print('Could not update', self.output_fname)

        pca = PCA(n_components=3)
        self.embedding = pca.fit_transform(self.cc_matrix)
        self._update_plot()

    def _update_plot(self):
        if self.embedding is None:
            print('Calculate CC matrix embedding first')
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.scatter(self.embedding[:,0], self.embedding[:,1],
                   c=self.embedding[:,2], cmap='coolwarm')
        ax.set_facecolor('k')
        self.fig.tight_layout()
        self.canvas.draw()

    def _get_nearest_class(self, event=None):
        if (self.embedding is None or
            # For newer matplotlib
            (hasattr(self.navbar, '_zoom_info') and self.navbar._zoom_info is not None) or
            # For matplotlib < 3.2
            (hasattr(self.navbar, '_zoom_mode') and self.navbar._zoom_mode is not None)):
            event.accept()
            return
        x = event.xdata
        y = event.ydata
        ax = event.inaxes

        nearest = np.linalg.norm(self.embedding[:,:2] - np.array([x,y]), axis=1).argmin()
        self.nearest_label.setText('Class %d' % nearest)
        cx, cy = self.embedding[nearest, :2]

        # Have to set bbox manually after plotting marker
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        if self._nearest_point is not None:
            self._nearest_point.remove()
        self._nearest_point = ax.plot(cx, cy,
                                      marker='x',
                                      color='lime',
                                      markersize=8)[0]
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        self.canvas.draw()

        self.parent.modenum.setValue(nearest)
        self.parent._modenum_changed()

    def closeEvent(self, event):
        self.windowClosed.emit()
        event.accept()

