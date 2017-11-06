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
import multiprocessing
import ctypes

class Manual_panel(QtWidgets.QWidget):
    def __init__(self, parent, *args, **kwargs):
        super(Manual_panel, self).__init__(parent, *args, **kwargs)
        
        self.setFixedWidth(280)
        self.parent = parent
        self.classes = self.parent.classes
        self.emc_reader = self.parent.emc_reader
        self.numstr = self.parent.frame_panel.numstr
        self.plot_frame = self.parent.frame_panel.plot_frame
        self.original_key_press = self.parent.keyPressEvent
        self.old_cnum = None
        
        self.init_UI()

    def init_UI(self):
        vbox = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel('Press any [a-z] key to assign label to frame', self)
        vbox.addWidget(label)

        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        self.classify_flag = QtWidgets.QCheckBox('Classify', self)
        self.classify_flag.setChecked(False)
        self.classify_flag.stateChanged.connect(self.classify_flag_changed)
        hbox.addWidget(self.classify_flag)
        button = QtWidgets.QPushButton('Unassign Class', self)
        button.clicked.connect(self.unassign_class)
        hbox.addWidget(button)
        hbox.addStretch(1)

        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        self.class_fname = QtWidgets.QLineEdit(self.classes.fname, self)
        self.class_fname.editingFinished.connect(self.update_name)
        hbox.addWidget(self.class_fname)
        button = QtWidgets.QPushButton('Save Classes', self)
        button.clicked.connect(self.classes.save)
        hbox.addWidget(button)
        hbox.addStretch(1)

        label = QtWidgets.QLabel('Classification Summary:', self)
        vbox.addWidget(label)
        self.class_list_summary = QtWidgets.QLabel('', self)
        self.class_list_summary.setText(self.classes.gen_summary())
        vbox.addWidget(self.class_list_summary)

        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        self.class_line = QtWidgets.QGridLayout()
        hbox.addLayout(self.class_line)
        self.refresh_class_line()
        hbox.addStretch(1)

        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        button = QtWidgets.QPushButton('Prev', self)
        button.clicked.connect(self.prev_frame)
        hbox.addWidget(button)
        button = QtWidgets.QPushButton('Next', self)
        button.clicked.connect(self.next_frame)
        hbox.addWidget(button)
        button = QtWidgets.QPushButton('Random', self)
        button.clicked.connect(self.rand_frame)
        hbox.addWidget(button)
        button = QtWidgets.QPushButton('Refresh', self)
        button.clicked.connect(self.refresh_class_line)
        hbox.addWidget(button)
        hbox.addStretch(1)

        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        button = QtWidgets.QPushButton('Class powder', self)
        button.clicked.connect(self.class_powder)
        hbox.addWidget(button)
        self.num_proc = QtWidgets.QLineEdit('4', self)
        self.num_proc.setFixedWidth(30)
        hbox.addWidget(self.num_proc)

        vbox.addStretch(1)

    def refresh_class_line(self):
        for i in reversed(range(self.class_line.count())):
            w = self.class_line.itemAt(i).widget()
            self.class_line.removeWidget(w)
            w.setParent(None)
        self.class_num = QtWidgets.QButtonGroup()
        button = QtWidgets.QRadioButton('All')
        button.setChecked(True)
        self.class_num.addButton(button, 0)
        self.class_line.addWidget(button, 0, 0)
        for i, k in enumerate(self.classes.key):
            if k == ' ':
                text = '  '
            else:
                text = k
            button = QtWidgets.QRadioButton(text, self)
            self.class_num.addButton(button, i+1)
            self.class_line.addWidget(button, (i+1)/5, (i+1)%5)
        self.class_list_summary.setText(self.classes.gen_summary())

    def assign_class(self, char):
        num = int(self.numstr.text())
        self.classes.clist[num] = char
        self.classes.unsaved = True
        self.class_list_summary.setText(self.classes.gen_summary())
        if self.class_line.count() != len(self.classes.key) + 1:
            self.refresh_class_line()
        self.next_frame()

    def unassign_class(self, event=None):
        num = int(self.numstr.text())
        self.classes.clist[num] = ' '
        self.classes.unsaved = True
        self.class_list_summary.setText(self.classes.gen_summary())
        if self.class_line.count() != len(self.classes.key) + 1:
            self.refresh_class_line()
        self.plot_frame()

    def classify_flag_changed(self, event=None):
        if self.classify_flag.isChecked():
            self.parent.keyPressEvent = self.classify_key_press
        else:
            self.parent.keyPressEvent = self.original_key_press

    def classify_key_press(self, event):
        if str(event.text()) in string.ascii_lowercase:
            self.assign_class(str(event.text()))

    def update_name(self, event=None):
        self.classes.fname = str(self.class_fname.text())

    def next_frame(self, event=None):
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
        
        if num < self.parent.num_frames:
            self.numstr.setText(str(num))
            self.plot_frame()

    def prev_frame(self, event=None):
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

    def rand_frame(self, event=None):
        cnum = self.class_num.checkedId() - 1
        if cnum == -1:
            num = np.random.randint(self.parent.num_frames)
        else:
            points = np.where(self.classes.key_pos == cnum)[0]
            num = points[np.random.randint(len(points))]
        self.numstr.setText(str(num))
        self.plot_frame()

    def class_powder(self, event=None):
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
            powders = multiprocessing.Array(ctypes.c_double, num_proc*self.parent.geom.mask.size)
            pshape = (num_proc,) + self.parent.geom.mask.shape 
            print 'Calculating powder sum for class %s using %d threads' % (self.class_num.checkedButton().text(), num_proc)
            jobs = []
            for i in range(num_proc):
                p = multiprocessing.Process(target=self.powder_worker, args=(i, points[i::num_proc], pshape, powders))
                jobs.append(p)
                p.start()
            for j in jobs:
                j.join()
            sys.stderr.write('\r%d/%d\n'%(len(points), len(points)))
            powder = np.frombuffer(powders.get_obj()).reshape(pshape).sum(0)
            self.class_powder = powder
            self.old_cnum = cnum
        self.plot_frame(frame=powder)

    def powder_worker(self, rank, my_points, pshape, powders):
        np_powder = np.frombuffer(powders.get_obj()).reshape(pshape)[rank]
        for i, p in enumerate(my_points):
            np_powder += self.emc_reader.get_frame(p)
            if rank == 0:
                sys.stderr.write('\r%d/%d'%(i,len(my_points)))

    def custom_hide(self):
        self.classify_flag.setChecked(False)
        self.classify_flag_changed()
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
        self.refresh_class_line()
        self.class_fname.setText(self.classes.fname)

