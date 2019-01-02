'''Module containing manual classification panel in the GUI'''

import sys
import string
import multiprocessing as mp
import ctypes
import numpy as np
try:
    from PyQt5 import QtWidgets # pylint: disable=import-error
except ImportError:
    import sip
    sip.setapi('QString', 2)
    from PyQt4 import QtGui as QtWidgets # pylint: disable=import-error
from . import gui_utils

class ManualPanel(QtWidgets.QWidget):
    '''Manual classification panel of the Classifier GUI
    Class names are [a-z], i.e. the lower case alphabet.
    In classification mode, pressing an alphabet key assigns the class and goes to the next frame.
    '''
    def __init__(self, parent, *args, **kwargs):
        super(ManualPanel, self).__init__(parent, *args, **kwargs)

        self.setFixedWidth(280)
        self.parent = parent
        self.classes = self.parent.classes
        self.emc_reader = self.parent.emc_reader
        self.numstr = self.parent.frame_panel.numstr
        self.plot_frame = self.parent.frame_panel.plot_frame
        self.original_key_press = self.parent.keyPressEvent
        self.old_cnum = self.class_powder = self.class_num = self.class_fname = None
        self.classify_flag = self.class_list_summary = self.class_line = None
        self.num_proc = None

        self._init_ui()

    def _init_ui(self):
        vbox = QtWidgets.QVBoxLayout(self)
        self._init_classify_ui(vbox)
        self._init_scroll_ui(vbox)
        vbox.addStretch(1)

    def _init_classify_ui(self, vbox):
        label = QtWidgets.QLabel('Press any [a-z] key to assign label to frame', self)
        vbox.addWidget(label)

        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        self.classify_flag = QtWidgets.QCheckBox('Classify', self)
        self.classify_flag.setChecked(False)
        self.classify_flag.stateChanged.connect(self._classify_flag_changed)
        hbox.addWidget(self.classify_flag)
        button = QtWidgets.QPushButton('Unassign Class', self)
        button.clicked.connect(self._unassign_class)
        hbox.addWidget(button)
        hbox.addStretch(1)

        gui_utils.add_class_hbox(self, vbox)

        label = QtWidgets.QLabel('Classification Summary:', self)
        vbox.addWidget(label)
        self.class_list_summary = QtWidgets.QLabel('', self)
        self.class_list_summary.setText(self.classes.gen_summary())
        vbox.addWidget(self.class_list_summary)

        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        self.class_line = QtWidgets.QGridLayout()
        hbox.addLayout(self.class_line)
        self._refresh_class_line()
        hbox.addStretch(1)

    def _init_scroll_ui(self, vbox):
        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        gui_utils.add_scroll_hbox(self, hbox)
        button = QtWidgets.QPushButton('Refresh', self)
        button.clicked.connect(self._refresh_class_line)
        hbox.addWidget(button)
        hbox.addStretch(1)

        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        button = QtWidgets.QPushButton('Class powder', self)
        button.clicked.connect(self._make_class_powder)
        hbox.addWidget(button)
        self.num_proc = QtWidgets.QLineEdit('4', self)
        self.num_proc.setFixedWidth(30)
        hbox.addWidget(self.num_proc)

    def _refresh_class_line(self):
        for i in reversed(range(self.class_line.count())):
            widg = self.class_line.itemAt(i).widget()
            self.class_line.removeWidget(widg)
            widg.setParent(None)
        self.class_num = QtWidgets.QButtonGroup()
        button = QtWidgets.QRadioButton('All')
        button.setChecked(True)
        self.class_num.addButton(button, 0)
        self.class_line.addWidget(button, 0, 0)
        for i, key in enumerate(self.classes.key):
            if key == ' ':
                text = '  '
            else:
                text = key
            button = QtWidgets.QRadioButton(text, self)
            self.class_num.addButton(button, i+1)
            self.class_line.addWidget(button, (i+1)/5, (i+1)%5)
        self.class_list_summary.setText(self.classes.gen_summary())

    def _assign_class(self, char):
        num = int(self.numstr.text())
        self.classes.clist[num] = char
        self.classes.unsaved = True
        self.class_list_summary.setText(self.classes.gen_summary())
        if self.class_line.count() != len(self.classes.key) + 1:
            self._refresh_class_line()
        self._next_frame()

    def _unassign_class(self):
        num = int(self.numstr.text())
        self.classes.clist[num] = ' '
        self.classes.unsaved = True
        self.class_list_summary.setText(self.classes.gen_summary())
        if self.class_line.count() != len(self.classes.key) + 1:
            self._refresh_class_line()
        self.plot_frame()

    def _classify_flag_changed(self):
        if self.classify_flag.isChecked():
            self.parent.keyPressEvent = self._classify_key_press
        else:
            self.parent.keyPressEvent = self.original_key_press

    def _classify_key_press(self, event):
        if str(event.text()) in string.ascii_lowercase:
            self._assign_class(str(event.text()))

    def _update_name(self):
        self.classes.fname = str(self.class_fname.text())

    def _next_frame(self):
        num = int(self.numstr.text())
        cnum = self.class_num.checkedId() - 1
        if cnum == -1:
            num += 1
        else:
            points = np.where(self.classes.key_pos == cnum)[0]
            index = np.searchsorted(points, num, side='left')
            if num in points:
                index += 1
            if index > len(points) - 1:
                index = len(points) - 1
            num = points[index]

        if num < self.emc_reader.num_frames:
            self.numstr.setText(str(num))
            self.plot_frame()

    def _prev_frame(self):
        num = int(self.numstr.text())
        cnum = self.class_num.checkedId() - 1
        if cnum == -1:
            num -= 1
        else:
            points = np.where(self.classes.key_pos == cnum)[0]
            index = np.searchsorted(points, num, side='left') - 1
            if index < 0:
                index = 0
            num = points[index]

        if num > -1:
            self.numstr.setText(str(num))
            self.plot_frame()

    def _rand_frame(self):
        cnum = self.class_num.checkedId() - 1
        if cnum == -1:
            num = np.random.randint(self.emc_reader.num_frames)
        else:
            points = np.where(self.classes.key_pos == cnum)[0]
            num = points[np.random.randint(len(points))]
        self.numstr.setText(str(num))
        self.plot_frame()

    def _make_class_powder(self):
        cnum = self.class_num.checkedId() - 1
        if cnum == self.old_cnum:
            powder = self.class_powder
        elif cnum == -1:
            powder = self.emc_reader.get_powder()
            self.class_powder = powder
            self.old_cnum = cnum
        else:
            points = np.where(self.classes.key_pos == cnum)[0]
            num_proc = int(self.num_proc.text())
            powders = mp.Array(ctypes.c_double, num_proc*self.parent.geom.mask.size)
            pshape = (num_proc,) + self.parent.geom.mask.shape
            print('Calculating powder sum for class %s using %d threads' %
                  (self.class_num.checkedButton().text(), num_proc))
            jobs = []
            for i in range(num_proc):
                proc = mp.Process(target=self._powder_worker,
                                  args=(i, points[i::num_proc], pshape, powders))
                jobs.append(proc)
                proc.start()
            for job in jobs:
                job.join()
            sys.stderr.write('\r%d/%d\n'%(len(points), len(points)))
            powder = np.frombuffer(powders.get_obj()).reshape(pshape).sum(0)
            self.class_powder = powder
            self.old_cnum = cnum
        self.plot_frame(frame=powder)

    def _powder_worker(self, rank, my_points, pshape, powders):
        np_powder = np.frombuffer(powders.get_obj()).reshape(pshape)[rank]
        for i, num in enumerate(my_points):
            np_powder += self.emc_reader.get_frame(num)
            if rank == 0:
                sys.stderr.write('\r%d/%d'%(i, len(my_points)))

