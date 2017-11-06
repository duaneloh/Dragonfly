import numpy as np
import sys
import os
import string
try:
    from PyQt5 import QtCore, QtWidgets, QtGui
except ImportError:
    import sip
    sip.setapi('QString', 2)
    from PyQt4 import QtCore, QtGui
    from PyQt4 import QtGui as QtWidgets
import polar
import multiprocessing
import ctypes

class Conversion_panel(QtWidgets.QWidget):
    def __init__(self, parent, *args, **kwargs):
        super(Conversion_panel, self).__init__(parent, *args, **kwargs)
        
        self.setFixedWidth(280)
        self.parent = parent
        self.frame = self.parent.frame_panel
        self.emc_reader = self.parent.emc_reader
        self.indices = np.arange(0, 1000, dtype='i4')
        
        self.polar = None
        self.init_UI()
        self.remake_converter(replot=False)

    def init_UI(self):
        vbox = QtWidgets.QVBoxLayout(self)
        
        label = QtWidgets.QLabel('Convert to angular correlations', self)
        vbox.addWidget(label)

        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        label = QtWidgets.QLabel('R_min:', self)
        hbox.addWidget(label)
        self.r_min = QtWidgets.QLineEdit('16')
        self.r_min.setFixedWidth(40)
        self.r_min.editingFinished.connect(self.remake_converter)
        hbox.addWidget(self.r_min)
        label = QtWidgets.QLabel('R_max:', self)
        hbox.addWidget(label)
        self.r_max = QtWidgets.QLineEdit('80')
        self.r_max.setFixedWidth(40)
        self.r_max.editingFinished.connect(self.remake_converter)
        hbox.addWidget(self.r_max)
        hbox.addStretch(1)

        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        label = QtWidgets.QLabel('dR:', self)
        hbox.addWidget(label)
        self.delta_r = QtWidgets.QLineEdit('2')
        self.delta_r.setFixedWidth(40)
        self.delta_r.editingFinished.connect(self.remake_converter)
        hbox.addWidget(self.delta_r)
        label = QtWidgets.QLabel(u'd\u03b8:', self)
        hbox.addWidget(label)
        self.delta_ang = QtWidgets.QLineEdit('10')
        self.delta_ang.setFixedWidth(40)
        self.delta_ang.editingFinished.connect(self.remake_converter)
        hbox.addWidget(self.delta_ang)
        label = QtWidgets.QLabel('deg', self)
        hbox.addWidget(label)
        hbox.addStretch(1)

        button = QtWidgets.QPushButton('Update', self)
        button.clicked.connect(self.remake_converter)
        vbox.addWidget(button)

        label = QtWidgets.QLabel('Batch processing', self)
        vbox.addWidget(label)

        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        label = QtWidgets.QLabel('Frames:', self)
        hbox.addWidget(label)
        self.first_frame = QtWidgets.QLineEdit('0', self)
        self.first_frame.setFixedWidth(60)
        hbox.addWidget(self.first_frame)
        label = QtWidgets.QLabel('-', self)
        hbox.addWidget(label)
        self.last_frame = QtWidgets.QLineEdit('1000', self)
        self.last_frame.setFixedWidth(60)
        hbox.addWidget(self.last_frame)
        label = QtWidgets.QLabel('Class:')
        hbox.addWidget(label)
        self.class_chars = QtWidgets.QLineEdit('', self)
        self.class_chars.setFixedWidth(20)
        hbox.addWidget(self.class_chars)
        hbox.addStretch(1)

        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        button = QtWidgets.QPushButton('Process', self)
        button.clicked.connect(self.convert_frames)
        hbox.addWidget(button)
        self.num_proc = QtWidgets.QLineEdit('1', self)
        self.num_proc.setFixedWidth(30)
        hbox.addWidget(self.num_proc)
        hbox.addStretch(1)
        self.method = QtWidgets.QComboBox(self)
        hbox.addWidget(self.method)
        self.method.addItem('ang_corr_normed')
        self.method.addItem('ang_corr')
        self.method.addItem('polar')
        self.method.addItem('polar_normed')
        self.method.addItem('raw')

        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        self.save_flag = QtWidgets.QCheckBox('Save', self)
        self.save_flag.setChecked(True)
        hbox.addWidget(self.save_flag)
        fname = os.path.relpath(self.parent.output_folder + '/converted.npy')
        self.save_fname = QtWidgets.QLineEdit(fname, self)
        hbox.addWidget(self.save_fname)
        
        vbox.addStretch(1)

    def remake_converter(self, replot=True, event=None):
        self.polar = polar.Polar_converter(self.parent.geom.cx, 
                                           self.parent.geom.cy, 
                                           self.parent.geom.unassembled_mask,
                                           r_min = float(self.r_min.text()),
                                           r_max = float(self.r_max.text()),
                                           delta_r = float(self.delta_r.text()),
                                           delta_ang = float(self.delta_ang.text()))
        if replot:
            self.frame.plot_frame()

    def convert_frames(self, event=None):
        try:
            start = int(self.first_frame.text())
            end = int(self.last_frame.text())
            num_proc = int(self.num_proc.text())
        except ValueError:
            sys.stderr.write('Integers only for frame range and number of processors\n')
            return
        
        self.indices = np.arange(start, end, dtype='i4')
        clist = self.parent.classes.clist[start:end]
        if self.class_chars.text() != '':
            sel = np.array([clist==c for c in self.class_chars.text()]).any(axis=0)
            self.indices = self.indices[sel]
        if len(self.indices) == 0:
            sys.stderr.write('No frames of class %s in frame range\n'%self.class_chars.text())
            return
        else:
            sys.stderr.write('Converting %d frames with %d processors\n' % (len(self.indices), num_proc))
        
        arr = self.get_and_convert(0)
        converted = multiprocessing.Array(ctypes.c_double, arr.size*len(self.indices))
        jobs = []
        for i in range(num_proc):
            p = multiprocessing.Process(target=self.convert_worker, args=(i, num_proc, self.indices, arr.size, converted))
            jobs.append(p)
            p.start()
        for j in jobs:
            j.join()
        sys.stderr.write('\r%d/%d\n' % (len(self.indices), len(self.indices)))
        
        self.parent.converted = np.frombuffer(converted.get_obj()).reshape(len(self.indices), -1)
        if self.save_flag.isChecked():
            sys.stderr.write('Saving angular correlations to %s\n'%self.save_fname.text())
            np.save(self.save_fname.text(), self.parent.converted)

    def get_and_convert(self, num):
        return self.polar.convert(self.emc_reader.get_frame(num, raw=True), method=self.method.currentText())

    def convert_worker(self, rank, num_proc, indices, size, converted):
        my_ind = indices[rank::num_proc]
        np_converted = np.frombuffer(converted.get_obj())
        for i, ind in enumerate(my_ind):
            ang_ind = np.where(indices==ind)[0][0]
            np_converted[size*ang_ind:size*(ang_ind+1)] = self.get_and_convert(ind).flatten()
            if rank == 0:
                sys.stderr.write('\r%d/%d'%(i, len(indices)))

    def plot_converted_frame(self, event=None):
        pframe = self.polar.compute_polar(self.emc_reader.get_frame(self.frame.get_num(), raw=True))
        
        fig = self.frame.fig
        s = fig.add_subplot(121)
        sc = fig.add_subplot(122)
        sc.imshow(pframe, vmin=0, vmax=float(self.frame.rangestr.text()), interpolation='none', cmap=self.parent.cmap, aspect=float(pframe.shape[1])/pframe.shape[0])
        sc.set_title('Polar representation')
        fig.add_subplot(sc)
        
        return s

    def custom_hide(self):
        r = self.parent.geometry()
        rp = self.geometry()
        r.setWidth(r.width() - rp.width())
        self.hide()
        self.parent.setGeometry(r)

    def custom_show(self):
        r = self.parent.geometry()
        rp = self.geometry()
        r.setWidth(r.width() + rp.width())
        self.parent.setGeometry(r)
        self.show()

