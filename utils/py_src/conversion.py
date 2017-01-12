import numpy as np
import sys
import os
import string
#import sip
#sip.setapi('Qstring', 2)
#from PyQt4 import QtGui
#from PyQt4 import QtCore
import polar
from PyQt5 import QtCore, QtGui, QtWidgets

class Conversion_panel(QtWidgets.QWidget):
    def __init__(self, parent, *args, **kwargs):
        super(Conversion_panel, self).__init__(parent, *args, **kwargs)
        
        self.setFixedWidth(280)
        self.parent = parent
        self.emc_reader = self.parent.emc_reader
        self.plot_frame = self.parent.frame_panel.plot_frame
        
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
        label = QtWidgets.QLabel('Frame range:', self)
        hbox.addWidget(label)
        self.first_frame = QtWidgets.QLineEdit('0')
        self.first_frame.setFixedWidth(64)
        hbox.addWidget(self.first_frame)
        label = QtWidgets.QLabel('-', self)
        hbox.addWidget(label)
        self.last_frame = QtWidgets.QLineEdit('1000')
        self.last_frame.setFixedWidth(64)
        hbox.addWidget(self.last_frame)
        hbox.addStretch(1)
        
        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        button = QtWidgets.QPushButton('Process', self)
        button.clicked.connect(self.convert_frames)
        hbox.addWidget(button)
        self.num_proc = QtWidgets.QLineEdit('1', self)
        self.num_proc.setFixedWidth(24)
        hbox.addWidget(self.num_proc)
        hbox.addStretch(1)
        self.save_flag = QtWidgets.QCheckBox('Save', self)
        self.save_flag.setChecked(True)
        hbox.addWidget(self.save_flag)
        self.normed_flag = QtWidgets.QCheckBox('Normed', self)
        self.normed_flag.setChecked(True)
        hbox.addWidget(self.normed_flag)
        
        vbox.addStretch(1)

    def remake_converter(self, replot=True, event=None):
        self.polar = polar.Polar_converter(self.parent.geom.cx, 
                                           self.parent.geom.cy, 
                                           self.parent.geom.raw_mask,
                                           r_min = float(self.r_min.text()),
                                           r_max = float(self.r_max.text()),
                                           delta_r = float(self.delta_r.text()),
                                           delta_ang = float(self.delta_ang.text()))
        if replot:
            self.plot_frame()

    def convert_frames(self, event=None):
        ang_corr = []
        try:
            start = int(self.first_frame.text())
            end = int(self.last_frame.text())
            num_proc = int(self.num_proc.text())
        except ValueError:
            print 'Integers only'
            return
        
        arr = self.get_and_convert(0)
        ang_corr = multiprocessing.Array(ctypes.c_double, arr.size*(end-start))
        jobs = []
        for i in range(num_proc):
            p = multiprocessing.Process(target=self.convert_worker, args=(i, num_proc, np.arange(start, end, dtype='i4'), arr.size, ang_corr))
            jobs.append(p)
            p.start()
        for j in jobs:
            j.join()
        sys.stderr.write('\r%d/%d\n' % (end, end))
        
        self.parent.ang_corr = np.frombuffer(ang_corr.get_obj()).reshape(end-start, -1)
        if self.save_flag.isChecked():
            fname = self.parent.output_folder + '/ang_corr.npy'
            print 'Saving angular correlations to', fname
            np.save(fname, self.parent.ang_corr)

    def get_and_convert(self, num):
        return self.polar.compute_ang_corr(self.polar.convert(self.emc_reader.get_frame(num)), normed=self.normed_flag.isChecked())

    def convert_worker(self, rank, num_proc, indices, size, ang_corr):
        my_ind = indices[rank::num_proc]
        np_ang_corr = np.frombuffer(ang_corr.get_obj())
        for i in my_ind:
            ang_ind = np.where(indices==i)[0][0]
            np_ang_corr[size*ang_ind:size*(ang_ind+1)] = self.get_and_convert(i).flatten()
            if rank == 0:
                sys.stderr.write('\r%d/%d'%(i, indices[-1]))

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

