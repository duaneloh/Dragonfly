import numpy as np
import sys
import os
import string
try:
    from PyQt5 import QtCore, QtWidgets, QtGui
    from matplotlib.backends.backend_qt5agg import FigureCanvas
except ImportError:
    import sip
    sip.setapi('QString', 2)
    from PyQt4 import QtCore, QtGui
    from PyQt4 import QtGui as QtWidgets
    from matplotlib.backends.backend_qt4agg import FigureCanvas
from matplotlib.figure import Figure
import slices

class Frame_panel(QtWidgets.QWidget):
    def __init__(self, parent, compare=False, powder=False, *args, **kwargs):
        super(Frame_panel, self).__init__(parent, *args, **kwargs)
        
        self.parent = parent
        self.emc_reader = self.parent.emc_reader
        self.num_frames = self.parent.num_frames
        self.do_compare = compare
        self.do_powder = powder
        if self.do_compare:
            self.slices = slices.Slice_generator(self.parent.geom, 'data/quat.dat')
        if self.do_powder:
            self.powder_sum = self.emc_reader.get_powder()
        
        self.numstr = '0'
        self.rangestr = '10'
        
        self.init_UI()

    def init_UI(self):
        vbox = QtWidgets.QVBoxLayout(self)
        
        self.fig = Figure(figsize=(6, 6))
        self.fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.05)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.mpl_connect('button_press_event', self.frame_focus)
        vbox.addWidget(self.canvas)
        
        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        if not self.do_powder:
            label = QtWidgets.QLabel('Frame number: ', self)
            hbox.addWidget(label)
            self.numstr = QtWidgets.QLineEdit('0', self)
            self.numstr.setFixedWidth(64)
            hbox.addWidget(self.numstr)
            label = QtWidgets.QLabel('/%d'%self.num_frames, self)
            hbox.addWidget(label)
        hbox.addStretch(1)
        if not self.do_powder and self.do_compare:
            self.compare_flag = QtWidgets.QCheckBox('Compare', self)
            self.compare_flag.clicked.connect(self.compare_flag_changed)
            hbox.addWidget(self.compare_flag)
        label = QtWidgets.QLabel('PlotMax:', self)
        hbox.addWidget(label)
        self.rangestr = QtWidgets.QLineEdit('10', self)
        self.rangestr.setFixedWidth(48)
        hbox.addWidget(self.rangestr)
        
        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        button = QtWidgets.QPushButton('Plot', self)
        button.clicked.connect(self.plot_frame)
        hbox.addWidget(button)
        if self.do_powder:
            button = QtWidgets.QPushButton('Save', self)
            button.clicked.connect(self.save_powder)
            hbox.addWidget(button)
        else:
            button = QtWidgets.QPushButton('Prev', self)
            button.clicked.connect(self.prev_frame)
            hbox.addWidget(button)
            button = QtWidgets.QPushButton('Next', self)
            button.clicked.connect(self.next_frame)
            hbox.addWidget(button)
            button = QtWidgets.QPushButton('Random', self)
            button.clicked.connect(self.rand_frame)
            hbox.addWidget(button)
        hbox.addStretch(1)
        button = QtWidgets.QPushButton('Quit', self)
        button.clicked.connect(self.parent.close)
        hbox.addWidget(button)
        
        self.show()
        if not self.do_compare:
            self.plot_frame()

    def plot_frame(self, event=None, frame=None):
        if self.parent.mode_val is not None:
            mode = self.parent.mode_val
        else:
            mode = None
        
        if frame is not None:
            pass
        elif self.do_powder:
            frame = self.powder_sum
        else:
            num = self.get_num()
            frame = self.emc_reader.get_frame(num)
        
        try:
            for p in self.parent.embedding_panel.roi_list:
                p.remove()
        except (ValueError, AttributeError):
            pass
        
        self.fig.clear()
        if mode == 2:
            s = self.parent.conversion_panel.plot_converted_frame()
        elif self.do_compare and self.compare_flag.isChecked():
            s = self.plot_slice(num)
        else:
            s = self.fig.add_subplot(111)
        s.imshow(frame, vmin=0, vmax=float(self.rangestr.text()), interpolation='none', cmap=self.parent.cmap)
        title = '%d photons' % frame.sum()
        if frame is None and (mode == 1 or mode == 3):
            title += ' (%s)' % self.parent.classes.clist[num]
        if mode == 4 and self.parent.mlp_panel.predictions is not None:
            title += ' [%s]' % self.parent.mlp_panel.predictions[num]
        if mode is None and not self.do_powder and self.parent.blacklist is not None and self.parent.blacklist[num] == 1:
            title += ' (bad frame)'
        s.set_title(title)
        self.fig.add_subplot(s)
        self.canvas.draw()

    def get_num(self, raw=False, event=None):
        try:
            num = int(self.numstr.text())
        except ValueError:
            sys.stderr.write('Frame number must be integer\n')
            return
        if num < 0 or num >= self.num_frames:
            sys.stderr.write('Frame number %d out of range!\n' % num)
            return
        return num

    def plot_slice(self, num, event=None):
        with open(self.parent.log_fname, 'r') as f:
            line = f.readlines()[-1]
            try:
                iteration = int(line.split()[0])
            except (IndexError, ValueError):
                sys.stderr.write('Unable to determine iteration number from %s\n' % self.parent.log_fname)
                sys.stderr.write('%s\n' % line)
                iteration = None
        
        if iteration > 0:
            s = self.fig.add_subplot(121)
            sc = self.fig.add_subplot(122)
            tomo = self.slices.get_slice(iteration, num)
            sc.imshow(tomo, cmap=self.parent.cmap, vmin=0, vmax=float(self.rangestr.text()), interpolation='none')
            self.fig.add_subplot(sc)
        else:
            s = self.fig.add_subplot(111)
        
        return s

    def next_frame(self, event=None):
        num = int(self.numstr.text()) + 1
        if num < self.num_frames:
            self.numstr.setText(str(num))
            self.plot_frame()

    def prev_frame(self, event=None):
        num = int(self.numstr.text()) - 1
        if num > -1:
            self.numstr.setText(str(num))
            self.plot_frame()

    def rand_frame(self, event=None):
        num = np.random.randint(0, self.num_frames)
        self.numstr.setText(str(num))
        self.plot_frame()

    def compare_flag_changed(self):
        self.plot_frame()

    def frame_focus(self, event=None):
        self.setFocus()

    def save_powder(self):
        fname = '%s/assem_powder.bin' % self.parent.output_folder
        sys.stderr.write('Saving assembled powder sum with shape %s to %s\n' % ((self.powder_sum.shape,), fname))
        self.powder_sum.tofile(fname)
        
        raw_powder = self.emc_reader.get_powder(raw=True)
        fname = '%s/powder.bin' % self.parent.output_folder
        sys.stderr.write('Saving assembled powder sum with shape %s to %s\n' % ((raw_powder.shape,), fname))
        raw_powder.tofile(fname)

    def keyPressEvent(self, event):
        k = event.key()
        
        if k == QtCore.Qt.Key_Return or k == QtCore.Qt.Key_Enter:
            self.plot_frame()
        elif k == QtCore.Qt.Key_Right or k == QtCore.Qt.Key_Down:
            self.next_frame()
        elif k == QtCore.Qt.Key_Left or k == QtCore.Qt.Key_Up:
            self.prev_frame()
        else:
            event.ignore()

