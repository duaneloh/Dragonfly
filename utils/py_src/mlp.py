import numpy as np
import sys
import os
import string
from sklearn import neural_network
import multiprocessing
import ctypes
try:
    from PyQt5 import QtCore, QtWidgets, QtGui
except ImportError:
    import sip
    sip.setapi('QString', 2)
    from PyQt4 import QtCore, QtGui
    from PyQt4 import QtGui as QtWidgets

class MLP_panel(QtWidgets.QWidget):
    def __init__(self, parent, *args, **kwargs):
        super(MLP_panel, self).__init__(parent, *args, **kwargs)
        
        self.setFixedWidth(240)
        self.parent = parent
        self.classes = self.parent.classes
        self.emc_reader = self.parent.emc_reader
        self.conversion = self.parent.conversion_panel
        
        self.predictions = None
        self.trained = False
        
        self.init_UI()
        self.remake_mlp()

    def init_UI(self):
        self.vbox = QtWidgets.QVBoxLayout(self)
        
        label = QtWidgets.QLabel('Multi-layer Perceptron', self)
        self.vbox.addWidget(label)
        
        hbox = QtWidgets.QHBoxLayout()
        self.vbox.addLayout(hbox)
        label = QtWidgets.QLabel('Hidden layer sizes', self)
        hbox.addWidget(label)
        self.layer_sizes = QtWidgets.QLineEdit('10, 10', self)
        self.layer_sizes.setFixedWidth(80)
        self.layer_sizes.editingFinished.connect(self.remake_mlp)
        hbox.addWidget(self.layer_sizes)
        
        hbox = QtWidgets.QHBoxLayout()
        self.vbox.addLayout(hbox)
        label = QtWidgets.QLabel('alpha', self)
        hbox.addWidget(label)
        self.alpha_var = QtWidgets.QLineEdit('0.1', self)
        self.alpha_var.setFixedWidth(80)
        self.alpha_var.editingFinished.connect(self.remake_mlp)
        hbox.addWidget(self.alpha_var)
        
        hbox = QtWidgets.QHBoxLayout()
        self.vbox.addLayout(hbox)
        button = QtWidgets.QPushButton('Update', self)
        button.clicked.connect(self.remake_mlp)
        hbox.addWidget(button)
        hbox.addStretch(1)
        
        hbox = QtWidgets.QHBoxLayout()
        self.vbox.addLayout(hbox)
        button = QtWidgets.QPushButton('Train', self)
        button.clicked.connect(self.train)
        hbox.addWidget(button)
        hbox.addStretch(1)
        
        self.vbox.addStretch(1)

    def remake_mlp(self, event=None):
        sizes = tuple([int(s.strip()) for s in str(self.layer_sizes.text()).split(',')])
        alpha = float(self.alpha_var.text())
        self.mlp = neural_network.MLPClassifier(hidden_layer_sizes=sizes, alpha=alpha)

    def train(self, event=None):
        self.get_training_data()
        if len(self.train_data) == 0:
            sys.stderr.write('Need to classify some of the converted frames in order to train\n')
            return
        self.mlp.fit(self.train_data, self.train_labels)
        sys.stderr.write('Done training\n')
        #print 'Score on training set =', self.mlp.score(self.train_data, self.train_labels)
        if not self.trained:
            self.add_predict_frame()
        self.trained = True

    def get_training_data(self):
        converted = self.parent.converted
        if converted is None:
            #self.conversion.convert_frames()
            self.parent.converted = np.load(self.parent.output_folder+'/converted.npy') #FIXME For debugging
            converted = self.parent.converted
        
        key_pos = self.classes.key_pos[int(self.conversion.first_frame.text()):int(self.conversion.last_frame.text())]
        self.train_data = converted[key_pos>0]
        self.train_labels = key_pos[key_pos>0]

    def add_predict_frame(self):
        self.vbox.setStretch(self.vbox.count()-1, 0)
        
        hbox = QtWidgets.QHBoxLayout()
        self.vbox.addLayout(hbox)
        self.predict_first = QtWidgets.QLineEdit('0')
        self.predict_first.setFixedWidth(64)
        hbox.addWidget(self.predict_first)
        label = QtWidgets.QLabel('-', self)
        hbox.addWidget(label)
        self.predict_last = QtWidgets.QLineEdit('1000')
        self.predict_last.setFixedWidth(64)
        hbox.addWidget(self.predict_last)
        hbox.addStretch(1)
        
        hbox = QtWidgets.QHBoxLayout()
        self.vbox.addLayout(hbox)
        button = QtWidgets.QPushButton('Predict', self)
        button.clicked.connect(self.predict)
        hbox.addWidget(button)
        self.num_proc = QtWidgets.QLineEdit('1', self)
        self.num_proc.setFixedWidth(24)
        hbox.addWidget(self.num_proc)
        hbox.addStretch(1)
        
        self.predict_summary = QtWidgets.QLabel('', self)
        self.gen_predict_summary()
        self.vbox.addWidget(self.predict_summary)
        
        hbox = QtWidgets.QHBoxLayout()
        self.vbox.addLayout(hbox)
        self.predictions_fname = QtWidgets.QLineEdit('predictions.dat', self)
        hbox.addWidget(self.predictions_fname)
        button = QtWidgets.QPushButton('Save', self)
        button.clicked.connect(self.save_predictions)
        hbox.addWidget(button)
        
        self.vbox.addStretch(1)

    def predict(self, event=None):
        try:
            first = int(self.predict_first.text())
            last = int(self.predict_last.text())
            num_proc = int(self.num_proc.text())
        except ValueError:
            sys.stderr.write('Integers only\n')
            return
        
        if last < 0:
            last = self.parent.num_frames
        if self.get_and_convert(first).shape[0] != self.parent.converted.shape[1]:
            sys.stderr.write('Wrong length for converted image (expected %d, got %d). You may need to update converter.\n' %
                (self.parent.converted.shape[1], self.get_and_convert(first).shape[0]))
            return
        
        predictions = multiprocessing.Array(ctypes.c_char, self.parent.num_frames)
        jobs = []
        for i in range(num_proc):
            p = multiprocessing.Process(target=self.predict_worker, args=(i, num_proc, np.arange(first, last, dtype='i4'), predictions))
            jobs.append(p)
            p.start()
        for j in jobs:
            j.join()
        sys.stderr.write('\r%d/%d\n' % (last, last))
        
        self.predictions = np.frombuffer(predictions.get_obj(), dtype='S1')
        self.gen_predict_summary()

    def get_and_convert(self, num):
        return self.conversion.polar.convert(self.emc_reader.get_frame(num, raw=True), method=self.conversion.method.currentText()).flatten()

    def predict_worker(self, rank, num_proc, indices, predictions):
        my_ind = indices[rank::num_proc]
        for i in my_ind:
            converted = np.expand_dims(self.get_and_convert(i), axis=0)
            predictions[i] = self.classes.key[self.mlp.predict(converted)[0]]
            if rank == 0:
                sys.stderr.write('\r%d/%d'%(i, indices[-1]))

    def gen_predict_summary(self, event=None):
        summary=''
        key, counts = np.unique(self.predictions, return_counts=True)
        for i, c in zip(key, counts):
            summary += '|%-4s|%7d|\n' % (i, c)
        self.predict_summary.setText(summary)

    def save_predictions(self, event=None):
        sys.stderr.write('Saving predictions list to %s\n'%str(self.predictions_fname.text()))
        np.savetxt(str(self.predictions_fname.text()), self.predictions, fmt='%s')

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

