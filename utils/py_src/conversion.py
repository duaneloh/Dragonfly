'''Module containing conversion panel class in the Classifier GUI'''

import sys
import os
import multiprocessing
import ctypes
import numpy as np
try:
    from PyQt5 import QtWidgets # pylint: disable=import-error
except ImportError:
    import sip
    sip.setapi('QString', 2)
    from PyQt4 import QtGui as QtWidgets # pylint: disable=import-error
from . import polar

class ConversionPanel(QtWidgets.QWidget):
    '''Conversion panel in Classifier GUI
    
    Can choose converter parameters, r_min, r_max, delta_r, delta_ang
    Also used to batch-convert many frames using multiple threads
    
    Methods:
        convert_frames(): Convert frames with paremeters specified in GUI
        plot_converted_frame(): Add subplot showing polar representation to frame panel
    '''
    def __init__(self, parent, *args, **kwargs):
        super(ConversionPanel, self).__init__(parent, *args, **kwargs)

        self.polar_params = self.proc_params = self.save_params = {}

        self.setFixedWidth(280)
        self.parent = parent
        self.frame = self.parent.frame_panel
        self.emc_reader = self.parent.emc_reader
        self.indices = np.arange(1000, dtype='i4')

        self.polar = None
        self.converted = None
        self._init_ui()
        self._remake_converter(replot=False)

    def _init_ui(self):
        vbox = QtWidgets.QVBoxLayout(self)
        self._init_ui_params(vbox)
        self._init_ui_batch(vbox)
        vbox.addStretch(1)

    def _init_ui_params(self, vbox):
        label = QtWidgets.QLabel('Convert to polar representation', self)
        vbox.addWidget(label)

        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        label = QtWidgets.QLabel('R_min:', self)
        hbox.addWidget(label)
        self.polar_params['r_min'] = QtWidgets.QLineEdit(self.parent.polar_params[0])
        self.polar_params['r_min'].setFixedWidth(40)
        self.polar_params['r_min'].returnPressed.connect(self._remake_converter)
        hbox.addWidget(self.polar_params['r_min'])
        label = QtWidgets.QLabel('R_max:', self)
        hbox.addWidget(label)
        self.polar_params['r_max'] = QtWidgets.QLineEdit(self.parent.polar_params[1])
        self.polar_params['r_max'].setFixedWidth(40)
        self.polar_params['r_max'].returnPressed.connect(self._remake_converter)
        hbox.addWidget(self.polar_params['r_max'])
        hbox.addStretch(1)

        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        label = QtWidgets.QLabel('dR:', self)
        hbox.addWidget(label)
        self.polar_params['delta_r'] = QtWidgets.QLineEdit(self.parent.polar_params[2])
        self.polar_params['delta_r'].setFixedWidth(40)
        self.polar_params['delta_r'].returnPressed.connect(self._remake_converter)
        hbox.addWidget(self.polar_params['delta_r'])
        label = QtWidgets.QLabel(u'd\u03b8:', self)
        hbox.addWidget(label)
        self.polar_params['delta_ang'] = QtWidgets.QLineEdit(self.parent.polar_params[3])
        self.polar_params['delta_ang'].setFixedWidth(40)
        self.polar_params['delta_ang'].returnPressed.connect(self._remake_converter)
        hbox.addWidget(self.polar_params['delta_ang'])
        label = QtWidgets.QLabel('deg', self)
        hbox.addWidget(label)
        hbox.addStretch(1)

        button = QtWidgets.QPushButton('Update', self)
        button.clicked.connect(self._remake_converter)
        vbox.addWidget(button)

    def _init_ui_batch(self, vbox):
        label = QtWidgets.QLabel('Batch processing', self)
        vbox.addWidget(label)

        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        label = QtWidgets.QLabel('Frames:', self)
        hbox.addWidget(label)
        self.proc_params['first_frame'] = QtWidgets.QLineEdit('0', self)
        self.proc_params['first_frame'].setFixedWidth(60)
        hbox.addWidget(self.proc_params['first_frame'])
        label = QtWidgets.QLabel('-', self)
        hbox.addWidget(label)
        self.proc_params['last_frame'] = QtWidgets.QLineEdit('1000', self)
        self.proc_params['last_frame'].setFixedWidth(60)
        hbox.addWidget(self.proc_params['last_frame'])
        label = QtWidgets.QLabel('Class:')
        hbox.addWidget(label)
        self.proc_params['class_chars'] = QtWidgets.QLineEdit('', self)
        self.proc_params['class_chars'].setFixedWidth(20)
        hbox.addWidget(self.proc_params['class_chars'])
        hbox.addStretch(1)

        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        button = QtWidgets.QPushButton('Process', self)
        button.clicked.connect(self.convert_frames)
        hbox.addWidget(button)
        self.proc_params['num_proc'] = QtWidgets.QLineEdit('1', self)
        self.proc_params['num_proc'].setFixedWidth(30)
        hbox.addWidget(self.proc_params['num_proc'])
        hbox.addStretch(1)
        self.proc_params['method'] = QtWidgets.QComboBox(self)
        hbox.addWidget(self.proc_params['method'])
        self.proc_params['method'].addItem('ang_corr_normed')
        self.proc_params['method'].addItem('ang_corr')
        self.proc_params['method'].addItem('polar')
        self.proc_params['method'].addItem('polar_normed')
        self.proc_params['method'].addItem('raw')
        self.proc_params['method'].addItem('raw_normed')

        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        self.save_params['save_flag'] = QtWidgets.QCheckBox('Save', self)
        self.save_params['save_flag'].setChecked(True)
        hbox.addWidget(self.save_params['save_flag'])
        fname = os.path.relpath(self.parent.output_folder + '/converted.npy')
        self.save_params['save_fname'] = QtWidgets.QLineEdit(fname, self)
        hbox.addWidget(self.save_params['save_fname'])

    def _remake_converter(self, replot=True):
        self.polar = polar.PolarConverter(self.parent.geom.cx,
                                          self.parent.geom.cy,
                                          self.parent.geom.unassembled_mask,
                                          r_min=float(self.polar_params['r_min'].text()),
                                          r_max=float(self.polar_params['r_max'].text()),
                                          delta_r=float(self.polar_params['delta_r'].text()),
                                          delta_ang=float(self.polar_params['delta_ang'].text()))
        if replot:
            self.frame.plot_frame()

    def convert_frames(self):
        '''Convert frames to specified type with given parameters
        Uses multiprocessing module for multi-threading
        Parameters of converter specified in GUI panel
        '''
        try:
            start = int(self.proc_params['first_frame'].text())
            end = int(self.proc_params['last_frame'].text())
            num_proc = int(self.proc_params['num_proc'].text())
        except ValueError:
            sys.stderr.write('Integers only for frame range and number of processors\n')
            return

        indices = np.arange(start, end, dtype='i4')
        clist = self.parent.classes.clist[start:end]
        if self.proc_params['class_chars'].text() != '':
            sel = np.array([clist == c for c in self.proc_params['class_chars'].text()]).any(axis=0)
            indices = indices[sel]
        if len(indices) == 0:
            sys.stderr.write('No frames of class %s in frame range\n'%
                             self.proc_params['class_chars'].text())
            return
        else:
            sys.stderr.write('Converting %d frames with %d processors\n' %
                             (len(indices), num_proc))

        arr = self._get_and_convert(0)
        converted = multiprocessing.Array(ctypes.c_double, arr.size*len(indices))
        jobs = []
        for i in range(num_proc):
            proc = multiprocessing.Process(target=self._convert_worker,
                                           args=([i, num_proc, arr.size], indices, converted))
            jobs.append(proc)
            proc.start()
        for job in jobs:
            job.join()
        sys.stderr.write('\r%d/%d\n' % (len(indices), len(indices)))

        self.converted = np.frombuffer(converted.get_obj()).reshape(len(indices), -1)
        self.indices = indices
        if self.save_params['save_flag'].isChecked():
            sys.stderr.write('Saving angular correlations to %s\n'%
                             self.save_params['save_fname'].text())
            np.save(self.save_params['save_fname'].text(), self.converted)

    def _get_and_convert(self, num):
        return np.array(self.polar.convert(self.emc_reader.get_frame(num, raw=True),
                                           method=self.proc_params['method'].currentText()))

    def _convert_worker(self, params, indices, converted):
        rank = params[0]
        num_proc = params[1]
        size = params[2]
        my_ind = indices[rank::num_proc]
        np_converted = np.frombuffer(converted.get_obj())
        for i, ind in enumerate(my_ind):
            ang_ind = np.where(indices == ind)[0][0]
            np_converted[size*ang_ind:size*(ang_ind+1)] = self._get_and_convert(ind).flatten()
            if rank == 0:
                sys.stderr.write('\r%d/%d'%(i, len(indices)))

    def plot_converted_frame(self):
        '''Adds subplot showing polar representation of frame
        Subplot shown on the right. The left half is returned for the caller to process
        Integrated in frame panel
        '''
        pframe = self.polar.compute_polar(self.emc_reader.get_frame(self.frame.get_num(), raw=True))

        fig = self.frame.fig
        sub = fig.add_subplot(121)
        subc = fig.add_subplot(122)
        subc.imshow(pframe, vmin=0, vmax=float(self.frame.rangestr.text()),
                    interpolation='none', cmap=self.parent.cmap,
                    aspect=float(pframe.shape[1])/pframe.shape[0])
        subc.set_title('Polar representation')
        fig.add_subplot(subc)

        return sub

