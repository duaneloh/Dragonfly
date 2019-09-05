'''Module containing frame plotting part of GUIs'''

import os
import sys
import numpy as np
try:
    from PyQt5 import QtGui, QtCore, QtWidgets # pylint: disable=import-error
    from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT #pylint: disable=no-name-in-module
except ImportError:
    import sip
    sip.setapi('QString', 2)
    from PyQt4 import QtGui, QtCore #pylint: disable=import-error
    from PyQt4 import QtGui as QtWidgets # pylint: disable=import-error
    from matplotlib.backends.backend_qt4agg import FigureCanvas, NavigationToolbar2QT #pylint: disable=no-name-in-module
import matplotlib
from matplotlib.figure import Figure
from matplotlib import colors
from . import slices
from . import gui_utils

class MyNavigationToolbar(NavigationToolbar2QT):
    def _icon(self, name):
        fname = os.path.abspath(os.path.dirname(__file__) + '/../../aux/icons/'+name) 
        pm = QtGui.QPixmap(fname)
        if hasattr(pm, 'setDevicePixelRatio'):
            pm.setDevicePixelRatio(self.canvas._dpi_ratio)
        return QtGui.QIcon(pm)

class FramePanel(QtWidgets.QWidget):
    '''GUI panel containing frame display widget
    
    Can scroll through frames of parent's EMCReader object

    Other parameters:
        compare - Side-by-side view of frames and best guess tomograms from reconstruction
        powder - Show sum of all frames

    Required members of parent class:
        emc_reader - Instance of EMCReader class
        geom - Instance of Detector class
        output_folder - (Only for compare mode) Folder with output data
        need_scaling - (Only for compare mode) Whether reconstruction was done with scaling
    '''
    def __init__(self, parent, compare=False, powder=False, noscroll=False, **kwargs):
        super(FramePanel, self).__init__(**kwargs)

        matplotlib.rcParams.update({
            'text.color': '#eff0f1',
            'xtick.color': '#eff0f1',
            'ytick.color': '#eff0f1',
            'axes.labelcolor': '#eff0f1',
            #'axes.facecolor': '#232629',
            #'figure.facecolor': '#232629'})
            'axes.facecolor': '#2a2a2f',
            'figure.facecolor': '#2a2a2f'})

        self.parent = parent
        self.emc_reader = self.parent.emc_reader
        self.do_compare = compare
        self.do_powder = powder
        self.noscroll = noscroll
        if self.do_compare:
            self.slices = slices.SliceGenerator(self.parent.geom, 'data/quat.dat',
                                                folder=self.parent.output_folder,
                                                need_scaling=self.parent.need_scaling)
        if self.do_powder:
            self.powder_sum = self.emc_reader.get_powder(raw=True)
        if self.parent.blacklist is not None:
            self.good_ind = np.where(self.parent.blacklist==0)[0]

        self.numstr = '0'
        self.rangestr = '10'
        self.skip_bad = None

        self._init_ui()

    def _init_ui(self):
        vbox = QtWidgets.QVBoxLayout(self)

        self.fig = Figure(figsize=(6, 6))
        self.fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.05)
        self.canvas = FigureCanvas(self.fig)
        self.navbar = MyNavigationToolbar(self.canvas, self)
        self.canvas.mpl_connect('button_press_event', self._frame_focus)
        vbox.addWidget(self.navbar)
        vbox.addWidget(self.canvas)

        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        if not self.do_powder:
            label = QtWidgets.QLabel('Frame number: ', self)
            hbox.addWidget(label)
            self.numstr = QtWidgets.QLineEdit('0', self)
            self.numstr.setFixedWidth(64)
            self.numstr.setToolTip('Frame number (0-indexed)')
            hbox.addWidget(self.numstr)
            label = QtWidgets.QLabel('/%d'%self.emc_reader.num_frames, self)
            label.setToolTip('Number of frames')
            hbox.addWidget(label)
        hbox.addStretch(1)
        if not self.do_powder and self.do_compare:
            self.compare_flag = QtWidgets.QCheckBox('Compare', self)
            self.compare_flag.clicked.connect(self._compare_flag_changed)
            self.compare_flag.setChecked(False)
            self.compare_flag.setToolTip('Show comparison of frame with most likely tomogram')
            hbox.addWidget(self.compare_flag)
        self.sym_flag = QtWidgets.QCheckBox('Symmetrize', self)
        self.sym_flag.clicked.connect(self.plot_frame)
        self.sym_flag.setChecked(False)
        self.sym_flag.setToolTip('Centro-symmetrize frame (only meaningful at small angles)')
        hbox.addWidget(self.sym_flag)
        label = QtWidgets.QLabel('PlotMax:', self)
        hbox.addWidget(label)
        self.rangestr = QtWidgets.QLineEdit('10', self)
        self.rangestr.setFixedWidth(48)
        self.rangestr.setToolTip('Value at which color scale maximizes')
        hbox.addWidget(self.rangestr)

        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        button = QtWidgets.QPushButton('Plot', self)
        button.clicked.connect(self.plot_frame)
        hbox.addWidget(button)
        if self.do_powder:
            button = QtWidgets.QPushButton('Save', self)
            button.clicked.connect(self._save_powder)
            button.setToolTip('Save powder sum data to file')
            hbox.addWidget(button)
        elif not self.noscroll:
            gui_utils.add_scroll_hbox(self, hbox)
            if self.parent.blacklist is not None:
                self.skip_bad = QtWidgets.QCheckBox('Skip bad', self)
                hbox.addWidget(self.skip_bad)
        hbox.addStretch(1)
        button = QtWidgets.QPushButton('Quit', self)
        button.clicked.connect(self.parent.close)
        hbox.addWidget(button)

        self.show()
        #if not self.do_compare:
        self.plot_frame()

    def plot_frame(self, event=None, frame=None):
        '''Update canvas according to GUI parameters
        Updated plot depends on mode (for classifier) and whether the GUI is in
        'compare' or 'powder' mode.
        '''
        try:
            mode = self.parent.mode_val
        except AttributeError:
            mode = None

        if frame is not None:
            num = None
            pass
        elif self.do_powder:
            det0 = self.emc_reader.flist[0]['geom']
            frame = det0.assemble_frame(self.powder_sum, zoomed=True, sym=self.sym_flag.isChecked())
            num = None
        else:
            num = self.get_num()
            if num is None:
                return
            frame = self.emc_reader.get_frame(num, zoomed=True, sym=self.sym_flag.isChecked())

        try:
            for point in self.parent.embedding_panel.roi_list:
                point.remove()
        except (ValueError, AttributeError):
            pass

        self.fig.clear()
        if mode == 2:
            subp = self.parent.conversion_panel.plot_converted_frame()
        elif self.do_compare and self.compare_flag.isChecked():
            subp = self._plot_slice(num)
        else:
            subp = self.fig.add_subplot(111)
        subp.imshow(frame, vmin=0, vmax=float(self.rangestr.text()),
                    interpolation='none', cmap=self.parent.cmap)
        subp.set_title(self._get_plot_title(frame, num, mode))
        self.fig.tight_layout()
        self.canvas.draw()

    def get_num(self):
        '''Get valid frame number from GUI
        Returns None if the types number is either unparseable or out of bounds
        '''
        try:
            num = int(self.numstr.text())
        except ValueError:
            sys.stderr.write('Frame number must be integer\n')
            return None
        if num < 0 or num >= self.emc_reader.num_frames:
            sys.stderr.write('Frame number %d out of range!\n' % num)
            return None
        return num

    def _plot_slice(self, num):
        with open(self.parent.log_fname, 'r') as fptr:
            line = fptr.readlines()[-1]
            try:
                iteration = int(line.split()[0])
            except (IndexError, ValueError):
                sys.stderr.write('Unable to determine iteration number from %s\n' %
                                 self.parent.log_fname)
                sys.stderr.write('%s\n' % line)
                iteration = None

        if iteration > 0:
            subp = self.fig.add_subplot(121)
            subpc = self.fig.add_subplot(122)
            tomo, info = self.slices.get_slice(iteration, num)
            vmax = float(self.rangestr.text())
            subpc.imshow(tomo, cmap=self.parent.cmap,
                         norm=colors.SymLogNorm(vmin=tomo.min(), linthresh=1e-2, vmax=vmax),
                         interpolation='none')
            subpc.set_title('Mutual Info. = %f'%info)
        else:
            subp = self.fig.add_subplot(111)

        return subp

    def _next_frame(self):
        if self.skip_bad is None or not self.skip_bad.isChecked():
            num = int(self.numstr.text()) + 1
        else:
            num = self.good_ind[np.searchsorted(self.good_ind, int(self.numstr.text())) + 1]
        if num < self.emc_reader.num_frames:
            self.numstr.setText(str(num))
            self.plot_frame()

    def _prev_frame(self):
        if self.skip_bad is None or not self.skip_bad.isChecked():
            num = int(self.numstr.text()) - 1
        else:
            num = self.good_ind[np.searchsorted(self.good_ind, int(self.numstr.text())) - 1]
        if num > -1:
            self.numstr.setText(str(num))
            self.plot_frame()

    def _rand_frame(self):
        if self.skip_bad is None or not self.skip_bad.isChecked():
            num = np.random.randint(0, self.emc_reader.num_frames)
        else:
            num = self.good_ind[np.random.randint(0, self.good_ind.size)]
        self.numstr.setText(str(num))
        self.plot_frame()

    def _get_plot_title(self, frame, num, mode):
        title = '%d photons' % frame.sum()
        if num is None:
            return title
        if frame is None and (mode == 1 or mode == 3):
            title += ' (%s)' % self.parent.classes.clist[num]
        if mode == 4 and self.parent.mlp_panel.predictions is not None:
            title += ' [%s]' % self.parent.mlp_panel.predictions[num]
        if (mode is None and
                not self.do_powder and
                self.parent.blacklist is not None and
                self.parent.blacklist[num] == 1):
            title += ' (bad frame)'
        return title

    def _compare_flag_changed(self):
        self.plot_frame()

    def _frame_focus(self, event): # pylint: disable=unused-argument
        self.setFocus()

    def _save_powder(self):
        fname = '%s/powder.bin' % self.parent.output_folder
        sys.stderr.write('Saving raw powder sum with shape %s to %s\n' %
                         ((self.powder_sum.shape,), fname))
        self.powder_sum.tofile(fname)

        det0 = self.emc_reader.flist[0]['geom']
        fr = det0.assemble_frame(self.powder_sum)
        fname = '%s/assem_powder.bin' % self.parent.output_folder
        sys.stderr.write('Saving assembled powder sum with shape %s to %s\n' %
                         ((fr.shape,), fname))
        fr.data.tofile(fname)

    def keyPressEvent(self, event): # pylint: disable=C0103
        '''Override of default keyPress event handler'''
        key = event.key()
        mod = int(event.modifiers())

        if QtGui.QKeySequence(mod+key) == QtGui.QKeySequence('Ctrl+N'):
            self._next_frame()
        elif QtGui.QKeySequence(mod+key) == QtGui.QKeySequence('Ctrl+P'):
            self._prev_frame()
        elif QtGui.QKeySequence(mod+key) == QtGui.QKeySequence('Ctrl+R'):
            self._rand_frame()
        elif key == QtCore.Qt.Key_Return or key == QtCore.Qt.Key_Enter:
            self.plot_frame()
        elif key == QtCore.Qt.Key_Right or key == QtCore.Qt.Key_Down:
            self._next_frame()
        elif key == QtCore.Qt.Key_Left or key == QtCore.Qt.Key_Up:
            self._prev_frame()
        else:
            event.ignore()

