import numpy as np
import sys
import os
import string
import matplotlib
import matplotlib.path
import matplotlib.patches
from sklearn import manifold
try:
    from PyQt5 import QtCore, QtWidgets, QtGui
except ImportError:
    import sip
    sip.setapi('QString', 2)
    from PyQt4 import QtCore, QtGui
    from PyQt4 import QtGui as QtWidgets

class Embedding_panel(QtWidgets.QWidget):
    def __init__(self, parent, *args, **kwargs):
        super(Embedding_panel, self).__init__(parent, *args, **kwargs)
        
        self.setFixedWidth(250)
        self.parent = parent
        self.classes = self.parent.classes
        self.conversion = self.parent.conversion_panel
        self.frame = self.parent.frame_panel
        self.plot_frame = self.frame.plot_frame
        self.canvas = self.frame.canvas
        self.numstr = self.frame.numstr
        
        self.positions = []
        self.roi_list = []
        self.path_list = []
        self.points_inside_list = []
        self.click_points_list = []
        self.embed = None
        self.roi_summary = None
        self.embedded = False
        
        self.init_UI()

    def init_UI(self):
        self.vbox = QtWidgets.QVBoxLayout(self)
        
        #label = QtWidgets.QLabel('Spectral manifold embedding', self)
        #self.vbox.addWidget(label)
        self.method = QtWidgets.QComboBox(self)
        self.vbox.addWidget(self.method)
        self.method.addItem('Spectral Embedding')
        self.method.addItem('Isomap')
        self.method.addItem('Modified LLE')
        self.method.addItem('Hessian LLE')
        self.method.addItem('Multi-dimensional Scaling')
        self.method.addItem('t-Stochastic Neighbor Embedding')
        
        hbox = QtWidgets.QHBoxLayout()
        self.vbox.addLayout(hbox)
        button = QtWidgets.QPushButton('Embed', self)
        button.clicked.connect(self.do_embedding)
        hbox.addWidget(button)
        self.track_flag = QtWidgets.QCheckBox('Draw ROI', self)
        self.track_flag.setChecked(False)
        self.track_flag.stateChanged.connect(self.track_flag_changed)
        hbox.addWidget(self.track_flag)
        hbox.addStretch(1)
        
        hbox = QtWidgets.QHBoxLayout()
        self.vbox.addLayout(hbox)
        label = QtWidgets.QLabel('X-axis:', self)
        hbox.addWidget(label)
        self.x_axis_num = QtWidgets.QLineEdit('0', self)
        self.x_axis_num.setFixedWidth(24)
        self.x_axis_num.editingFinished.connect(self.gen_hist)
        hbox.addWidget(self.x_axis_num)
        label = QtWidgets.QLabel('Y-axis:', self)
        hbox.addWidget(label)
        self.y_axis_num = QtWidgets.QLineEdit('1', self)
        self.y_axis_num.setFixedWidth(24)
        self.y_axis_num.editingFinished.connect(self.gen_hist)
        hbox.addWidget(self.y_axis_num)
        hbox.addStretch(1)
        
        self.vbox.addStretch(1)

    def do_embedding(self, event=None):
        converted = self.parent.converted
        if converted is None:
            #self.conversion.convert_frames()
            self.parent.converted = np.load(self.parent.output_folder+'/converted.npy') #FIXME For debugging
            converted = self.parent.converted
        
        method_ind = self.method.currentIndex()
        print('Doing %s' % self.method.currentText())
        if method_ind == 0:
            self.embedder = manifold.SpectralEmbedding(n_components=4, n_jobs=-1)
        elif method_ind == 1:
            self.embedder = manifold.Isomap(n_components=4, n_jobs=-1)
        elif method_ind == 2:
            self.embedder = manifold.LocallyLinearEmbedding(n_components=4, n_jobs=-1, n_neighbors=20, method='modified')
        elif method_ind == 3:
            self.embedder = manifold.LocallyLinearEmbedding(n_components=4, n_jobs=-1, n_neighbors=20, method='hessian', eigen_solver='dense')
        elif method_ind == 4:
            self.embedder = manifold.MDS(n_components=4, n_jobs=-1)
        elif method_ind == 5:
            self.embedder = manifold.TSNE(n_components=3, init='pca')
        self.embedder.fit(converted)
        self.embed = self.embedder.embedding_
        self.embed_plot = self.embed
        
        self.gen_hist()
        self.plot_embedding()
        if not self.embedded:
            self.add_classes_frame()
        self.embedded = True

    def plot_embedding(self, event=None):
        try:
            for p in self.roi_list:
                p.remove()
        except (ValueError, AttributeError) as e:
            pass
        
        fig = self.frame.fig
        fig.clear()
        s = fig.add_subplot(111)
        e = self.embed_plot
        try:
            xnum = int(self.x_axis_num.text())
            ynum = int(self.y_axis_num.text())
        except ValueError:
            sys.stderr.write('Need axes numbers to be integers\n')
            return
        s.hist2d(e[:,xnum], e[:,ynum], bins=[self.binx, self.biny], vmax=float(self.frame.rangestr.text()), cmap=self.parent.cmap)
        s.set_title(self.method.currentText())
        for p in self.roi_list:
            s.add_artist(p)
        fig.add_subplot(s)
        self.frame.canvas.draw()

    def gen_hist(self, event=None):
        try:
            xnum = int(self.x_axis_num.text())
            ynum = int(self.y_axis_num.text())
        except ValueError:
            sys.stderr.write('Need axes numbers to be integers\n')
            return
        self.hist2d, self.binx, self.biny = np.histogram2d(self.embed[:,xnum], self.embed[:,ynum], bins=100)
        
        delx = self.binx[1] - self.binx[0]
        dely = self.biny[1] - self.biny[0]
        self.binx = np.insert(self.binx, 0, [self.binx[0]-6*delx, self.binx[0]-delx])
        self.binx = np.insert(self.binx, len(self.binx), [self.binx[-1]+delx, self.binx[-1]+6*delx])
        self.biny = np.insert(self.biny, 0, [self.biny[0]-6*dely, self.biny[0]-dely])
        self.biny = np.insert(self.biny, len(self.biny), [self.biny[-1]+dely, self.biny[-1]+6*dely])

    def add_classes_frame(self):
        self.vbox.setStretch(self.vbox.count()-1, 0)
        
        hbox = QtWidgets.QHBoxLayout()
        self.vbox.addLayout(hbox)
        self.class_line = QtWidgets.QGridLayout()
        hbox.addLayout(self.class_line)
        hbox.addStretch(1)
        self.class_num = QtWidgets.QButtonGroup()
        self.refresh_classes()
        
        hbox = QtWidgets.QHBoxLayout()
        self.vbox.addLayout(hbox)
        button = QtWidgets.QPushButton('Show', self)
        button.clicked.connect(self.show_selected_class)
        hbox.addWidget(button)
        button = QtWidgets.QPushButton('See all', self)
        button.clicked.connect(self.show_all_classes)
        hbox.addWidget(button)
        button = QtWidgets.QPushButton('Refresh', self)
        button.clicked.connect(self.refresh_classes)
        hbox.addWidget(button)
        hbox.addStretch(1)
        
        self.vbox.addStretch(1)

    def show_selected_class(self, event=None):
        ind = self.conversion.indices
        key_pos = np.where(self.classes.key_pos[ind] == self.class_num.checkedId())[0]
        self.embed_plot = self.embed[key_pos]
        self.plot_embedding()

    def show_all_classes(self, event=None):
        self.embed_plot = self.embed
        self.plot_embedding()

    def refresh_classes(self,event=None):
        for i in reversed(range(self.class_line.count())):
            w = self.class_line.itemAt(i).widget()
            self.class_line.removeWidget(w)
            w.setParent(None)
        for i, k in enumerate(self.classes.key):
            if k == ' ':
                text = '  '
            else:
                text = k
            button = QtWidgets.QRadioButton(text, self)
            button.clicked.connect(self.show_selected_class)
            if i == 0:
                button.setChecked(True)
            self.class_num.addButton(button, i)
            self.class_line.addWidget(button, i/5, i%5)

    def track_flag_changed(self, event=None):
        if self.track_flag.isChecked():
            if self.embed is not None:
                self.connect_id = self.frame.canvas.mpl_connect('button_press_event', self.track_positions)
        else:
            self.end_track_positions()

    def track_positions(self, event=None):
        self.event = event
        x = event.xdata
        y = event.ydata
        self.click_points_list.append(self.frame.fig.get_axes()[0].plot([x], [y], marker='.', markersize=8., color='white')[0])
        self.frame.canvas.draw()
        self.positions.append([x, y])

    def end_track_positions(self):
        pos = np.array(self.positions)
        if pos.size == 0:
            return
        try:
            xnum = int(self.x_axis_num.text())
            ynum = int(self.y_axis_num.text())
        except ValueError:
            sys.stderr.write('Need axes numbers to be integers\n')
            return
        pos = np.append(pos, pos[-1]).reshape(-1,2)
        self.path_list.append(matplotlib.path.Path(pos, closed=True))
        points_inside = np.array([self.path_list[-1].contains_point((p[xnum], p[ynum])) for p in self.embed])
        sys.stderr.write('%d/%d frames inside ROI %d\n' % (points_inside.sum(), len(points_inside), len(self.points_inside_list)))
        self.points_inside_list.append(self.conversion.indices[np.where(points_inside)[0]])
        
        self.roi_list.append(
            matplotlib.patches.PathPatch(
                self.path_list[-1],
                color='white',
                fill=False,
                linewidth=2.,
                figure=self.frame.fig
            )
        )
        self.frame.fig.get_axes()[0].add_artist(self.roi_list[-1])
        for p in self.click_points_list:
            p.remove()
        self.frame.canvas.draw()
        
        self.frame.canvas.mpl_disconnect(self.connect_id)
        self.positions = []
        self.click_points_list = []
        if self.roi_summary is None:
            self.add_roi_frame()
        elif self.roi_summary.text() == '':
            self.roi_frame.show()
        self.gen_roi_summary()
        self.add_roi_radiobutton(len(self.roi_list)-1)

    def add_roi_frame(self):
        self.vbox.setStretch(self.vbox.count()-1, 0)
        self.roi_frame = QtWidgets.QFrame(self)
        self.vbox.addWidget(self.roi_frame)
        self.vbox.addStretch(1)
        vbox = QtWidgets.QVBoxLayout()
        self.roi_frame.setLayout(vbox)
        
        self.roi_summary = QtWidgets.QLabel('', self)
        vbox.addWidget(self.roi_summary)
        
        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        button = QtWidgets.QPushButton('Clear ROIs', self)
        button.clicked.connect(self.clear_roi)
        hbox.addWidget(button)
        hbox.addStretch(1)
        
        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        self.roi_choice = QtWidgets.QGridLayout()
        self.current_roi = QtWidgets.QButtonGroup()
        hbox.addLayout(self.roi_choice)
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
        hbox.addStretch(1)
        
        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        self.class_tag = QtWidgets.QLineEdit('', self)
        self.class_tag.setFixedWidth(24)
        hbox.addWidget(self.class_tag)
        button = QtWidgets.QPushButton('Apply Class', self)
        button.clicked.connect(self.apply_class)
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

    def add_roi_radiobutton(self, num):
        button = QtWidgets.QRadioButton(str(num))
        if num == 0:
            button.setChecked(True)
        self.current_roi.addButton(button, num)
        self.roi_choice.addWidget(button, num/5, num%5)

    def update_name(self, event=None):
        self.classes.fname = str(self.class_fname.text())

    def gen_roi_summary(self):
        summary = 'Embedded frames = %d\n' % len(self.embed)
        for i, p in enumerate(self.points_inside_list):
            summary += '%3d:%-5d ' % (i, len(p))
            if i%5 == 4:
                summary += '\n'
        self.roi_summary.setText(summary)

    def clear_roi(self):
        for p in self.roi_list:
            p.remove()
        self.frame.canvas.draw()
        self.roi_list = []
        self.path_list = []
        self.points_inside_list = []
        for i in reversed(range(self.roi_choice.count())):
            w = self.roi_choice.itemAt(i).widget()
            self.roi_choice.removeWidget(w)
            w.setParent(None)
        self.roi_frame.hide()
        self.roi_summary.setText('')

    def prev_frame(self, event=None):
        num = int(self.numstr.text())
        points = self.points_inside_list[self.current_roi.checkedId()]
        index = np.searchsorted(points, num, side='left') - 1
        if index < 0:
            index = 0
        self.numstr.setText(str(points[index]))
        self.plot_frame()

    def next_frame(self, event=None):
        num = int(self.numstr.text())
        points = self.points_inside_list[self.current_roi.checkedId()]
        index = np.searchsorted(points, num, side='left') + 1
        if index > len(points) - 1:
            index = len(points) - 1
        self.numstr.setText(str(points[index]))
        self.plot_frame()

    def rand_frame(self, event=None):
        points = self.points_inside_list[self.current_roi.checkedId()]
        self.numstr.setText(str(points[np.random.randint(len(points))]))
        self.plot_frame()

    def apply_class(self, event=None):
        roi_num = self.current_roi.checkedId()
        class_char = str(self.class_tag.text())
        self.classes.clist[self.points_inside_list[roi_num]] = class_char
        if self.class_line.count() != len(self.classes.key) + 1:
            self.refresh_classes()
        self.classes.gen_summary()
        self.classes.unsaved = True

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
        if self.embedded:
            self.refresh_classes()
        if self.roi_summary is not None:
            self.class_fname.setText(self.classes.fname)

