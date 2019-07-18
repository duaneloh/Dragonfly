'''Module containing MLP panel for the classifier GUI'''

from __future__ import print_function
import sys
import multiprocessing as mp
import ctypes
import numpy as np
try:
    from PyQt5 import QtWidgets # pylint: disable=import-error
except ImportError:
    import sip
    sip.setapi('QString', 2)
    from PyQt4 import QtGui as QtWidgets # pylint: disable=import-error
from sklearn import neural_network

class MLPPanel(QtWidgets.QWidget):
    '''Multi Layer Pereceptron panel for the classifier GUI
    
    This class takes some classified data and trains an MLP network
    This network can then be used to make predictions on the rest of the unclassified frames
    '''
    def __init__(self, parent, *args, **kwargs):
        super(MLPPanel, self).__init__(parent, *args, **kwargs)

        self.setFixedWidth(240)
        self.parent = parent
        self.classes = self.parent.classes
        self.emc_reader = self.parent.emc_reader
        self.conversion = self.parent.conversion_panel

        self.predictions = self.predict_first = self.predict_last = None
        self.predictions_fname = self.predict_summary = self.train_data = None
        self.train_labels = self.num_proc = None
        self.trained = False

        self._init_ui()
        self._remake_mlp()

    def _init_ui(self):
        self.vbox = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel('Multi-layer Perceptron', self)
        self.vbox.addWidget(label)

        hbox = QtWidgets.QHBoxLayout()
        self.vbox.addLayout(hbox)
        label = QtWidgets.QLabel('Hidden layer sizes', self)
        hbox.addWidget(label)
        self.layer_sizes = QtWidgets.QLineEdit('10, 10', self)
        self.layer_sizes.setFixedWidth(80)
        self.layer_sizes.editingFinished.connect(self._remake_mlp)
        hbox.addWidget(self.layer_sizes)

        hbox = QtWidgets.QHBoxLayout()
        self.vbox.addLayout(hbox)
        label = QtWidgets.QLabel('alpha', self)
        hbox.addWidget(label)
        self.alpha_var = QtWidgets.QLineEdit('0.1', self)
        self.alpha_var.setFixedWidth(80)
        self.alpha_var.editingFinished.connect(self._remake_mlp)
        hbox.addWidget(self.alpha_var)

        hbox = QtWidgets.QHBoxLayout()
        self.vbox.addLayout(hbox)
        button = QtWidgets.QPushButton('Update', self)
        button.clicked.connect(self._remake_mlp)
        hbox.addWidget(button)
        hbox.addStretch(1)

        hbox = QtWidgets.QHBoxLayout()
        self.vbox.addLayout(hbox)
        button = QtWidgets.QPushButton('Train', self)
        button.clicked.connect(self.train)
        hbox.addWidget(button)
        hbox.addStretch(1)

        self.vbox.addStretch(1)

    def _remake_mlp(self):
        sizes = tuple([int(s.strip()) for s in str(self.layer_sizes.text()).split(',')])
        alpha = float(self.alpha_var.text())
        self.mlp = neural_network.MLPClassifier(hidden_layer_sizes=sizes, alpha=alpha)

    def train(self):
        '''Train perceptron based on training data
        This data consists of a list of converted frames with a corresponding list of class labels
        After training, prediction options are enabled.
        '''
        self._get_training_data()
        if len(self.train_data) == 0:
            sys.stderr.write('Need to classify some of the converted frames in order to train\n')
            return
        print('Training with %d frames in the following classes:'%len(self.train_labels), self.classes.key[1:])
        self.mlp.fit(self.train_data, self.train_labels)
        print('Done training')
        #print('Score on training set =', self.mlp.score(self.train_data, self.train_labels))
        if not self.trained:
            self._add_predict_frame()
        self.trained = True

    def _get_training_data(self):
        converted = self.conversion.converted
        if converted is None:
            #self.conversion.convert_frames()
            self.conversion.converted = np.load(self.parent.output_folder+'/converted.npy')
            converted = self.conversion.converted

        first = int(self.conversion.proc_params['first_frame'].text())
        last = int(self.conversion.proc_params['last_frame'].text())
        key_pos = self.classes.key_pos[first:last]
        self.train_data = converted[key_pos > 0]
        self.train_labels = key_pos[key_pos > 0]

    def _add_predict_frame(self):
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
        self._gen_predict_summary()
        self.vbox.addWidget(self.predict_summary)

        hbox = QtWidgets.QHBoxLayout()
        self.vbox.addLayout(hbox)
        self.predictions_fname = QtWidgets.QLineEdit('predictions.dat', self)
        hbox.addWidget(self.predictions_fname)
        button = QtWidgets.QPushButton('Save', self)
        button.clicked.connect(self._save_predictions)
        hbox.addWidget(button)

        self.vbox.addStretch(1)

    def predict(self):
        '''Predict a range of frames using the trained classifier
        Even frames which are already classified will be predicted
        '''
        try:
            first = int(self.predict_first.text())
            last = int(self.predict_last.text())
            num_proc = int(self.num_proc.text())
        except ValueError:
            sys.stderr.write('Integers only\n')
            return

        if last < 0:
            last = self.emc_reader.num_frames
        if self._get_and_convert(first).shape[0] != self.conversion.converted.shape[1]:
            sys.stderr.write('Wrong length for converted image (expected %d, got %d).'
                             'You may need to update converter.\n' %
                             (self.conversion.converted.shape[1],
                              self._get_and_convert(first).shape[0]))
            return

        predictions = mp.Array(ctypes.c_char, int(self.emc_reader.num_frames))
        jobs = []
        for i in range(num_proc):
            proc = mp.Process(target=self._predict_worker,
                              args=(i, num_proc, np.arange(first, last, dtype='i4'), predictions))
            jobs.append(proc)
            proc.start()
        for job in jobs:
            job.join()
        sys.stderr.write('\r%d/%d\n' % (last, last))

        self.predictions = np.frombuffer(predictions.get_obj(), dtype='S1')
        self.predictions = np.array([key.decode('utf-8') for key in self.predictions])
        self._gen_predict_summary()

    def _get_and_convert(self, num):
        return self.conversion.polar.convert(self.emc_reader.get_frame(num, raw=True),
                                             method=self.conversion.proc_params['method'].currentText()).flatten()

    def _predict_worker(self, rank, num_proc, indices, predictions):
        my_ind = indices[rank::num_proc]
        for i in my_ind:
            converted = np.expand_dims(self._get_and_convert(i), axis=0)
            try:
                predictions[i] = bytes(self.classes.key[self.mlp.predict(converted)[0]], 'utf-8')
            except TypeError:  # Python 2
                predictions[i] = bytes(self.classes.key[self.mlp.predict(converted)[0]])
            if rank == 0:
                sys.stderr.write('\r%d/%d'%(i, indices[-1]))

    def _gen_predict_summary(self):
        summary = ''
        key, counts = np.unique(self.predictions, return_counts=True)
        for i, count in zip(key, counts):
            summary += '|%-4s|%7d|\n' % (i, count)
        self.predict_summary.setText(summary)

    def _save_predictions(self):
        sys.stderr.write('Saving predictions list to %s\n'%str(self.predictions_fname.text()))
        np.savetxt(str(self.predictions_fname.text()), self.predictions, fmt='%s')
