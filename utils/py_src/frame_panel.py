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
    def __init__(self, parent, *args, **kwargs):
        super(Frame_panel, self).__init__(parent, *args, **kwargs)
        
        self.parent = parent
        self.emc_reader = self.parent.emc_reader
        self.num_frames = self.parent.num_frames
        self.cmap = self.parent.cmap
        self.slices = slices.Slice_generator(self.parent.geom, 'data/quat.dat')
        
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
        label = QtWidgets.QLabel('Frame number: ', self)
        hbox.addWidget(label)
        self.numstr = QtWidgets.QLineEdit('0', self)
        self.numstr.setFixedWidth(64)
        hbox.addWidget(self.numstr)
        label = QtWidgets.QLabel('/%d'%self.num_frames, self)
        hbox.addWidget(label)
        hbox.addStretch(1)
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
        
        self.plot_frame()
        self.show()

    def plot_frame(self, event=None, embed=None, force_frame=False):
        if self.parent.mode_val is not None:
            mode = self.parent.mode_val
        else:
            mode = 0
        
        try:
            num = int(self.numstr.text())
        except ValueError:
            sys.stderr.write('Frame number must be integer\n')
            return
        
        if num < 0 or num >= self.num_frames:
            sys.stderr.write('Frame number %d out of range!\n' % num)
            return
        
        frame = self.emc_reader.get_frame(num)
        
        if mode == 2:
            # Conversion panel
            self.fig.clear()
            s = self.fig.add_subplot(121)
            s.imshow(frame, vmin=0, vmax=float(self.rangestr.text()), interpolation='none', cmap=self.cmap)
            s.set_title("%d photons" % frame.sum())
            self.fig.add_subplot(s)
            
            s = self.fig.add_subplot(122)
            pframe = self.parent.conversion_panel.polar.convert(frame)
            s.imshow(pframe, vmin=0, vmax=float(self.rangestr.text()), interpolation='none', cmap=self.cmap, aspect=float(pframe.shape[1])/pframe.shape[0])
            title = 'Polar representation'
            self.fig.add_subplot(s)
        elif (not force_frame) and mode == 3 and self.parent.embedding_panel.embed is not None:
            # Embedding 
            ep = self.parent.embedding_panel
            try:
                for p in ep.roi_list:
                    p.remove()
            except ValueError:
                pass
            
            self.fig.clear()
            s = self.fig.add_subplot(111)
            e = ep.embed_plot
            try:
                xnum = int(ep.x_axis_num.text())
                ynum = int(ep.y_axis_num.text())
            except ValueError:
                sys.stderr.write('Need axes numbers to be integers\n')
                return
            s.hist2d(e[:,xnum], e[:,ynum], bins=[ep.binx, ep.biny], vmax=float(self.rangestr.text()), cmap=self.cmap)
            title = 'Spectral embedding'
            for p in ep.roi_list:
                s.add_artist(p)
            self.fig.add_subplot(s)
        else:
            if mode == 3:
                try:
                    for p in self.parent.embedding_panel.roi_list:
                        p.remove()
                except ValueError:
                    pass
            
            self.fig.clear()
            if self.compare_flag.isChecked():
                with open('EMC.log', 'r') as f:
                    line = f.readlines()[-1]
                try:
                    iteration = int(line.split()[0])
                    s = self.fig.add_subplot(121)
                    sc = self.fig.add_subplot(122)
                    tomo = self.slices.get_slice(iteration, num)
                    sc.imshow(tomo, cmap=self.cmap, vmin=0, vmax=float(self.rangestr.text()), interpolation='none')
                    self.fig.add_subplot(sc)
                except (IndexError, ValueError):
                    sys.stderr.write('Unable to determine iteration number from EMC.log\n')
                    s = self.fig.add_subplot(111)
            else:
                s = self.fig.add_subplot(111)
            s.imshow(frame, vmin=0, vmax=float(self.rangestr.text()), interpolation='none', cmap=self.cmap)
            title = '%d photons' % frame.sum()
            if mode == 1:
                title += ' (%s)' % self.parent.classes.clist[num]
            if mode == 4 and self.parent.mlp_panel.predictions is not None:
                title += ' [%s]' % self.parent.mlp_panel.predictions[num]
            if self.parent.mode_val is None and self.parent.blacklist is not None and self.parent.blacklist[num] == 1:
                title += ' (bad frame)'
            s.set_title(title)
            self.fig.add_subplot(s)
        self.canvas.draw()

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

