'''Module containing embedding panel class in the Classifier GUI'''

from __future__ import print_function
import sys
import numpy as np
try:
    from PyQt5 import QtWidgets # pylint: disable=import-error
except ImportError:
    import sip
    sip.setapi('QString', 2)
    from PyQt4 import QtGui as QtWidgets # pylint: disable=import-error
import matplotlib
import matplotlib.path
import matplotlib.patches
from sklearn import manifold
from . import gui_utils

class EmbeddingPanel(QtWidgets.QWidget):
    '''Embedding panel in Classifier GUI
    Used to perform manifold embedding on converted data frames
    The frames are embedded in 4 dimensions, allowing for region selection
    A 2D histogram of the various points are shown, allowing the user to select
        a region of frames to either browse through or assign a class to.

    No public methods (all actions through GUI buttons)
    '''
    def __init__(self, parent, *args, **kwargs):
        super(EmbeddingPanel, self).__init__(parent, *args, **kwargs)

        self.setFixedWidth(250)
        self.parent = parent
        self.classes = self.parent.classes
        self.conversion = self.parent.conversion_panel
        self.frame = self.parent.frame_panel
        self.positions = []

        self.roi_list = []
        self.path_list = []
        self.points_inside_list = []
        self.click_points_list = []
        self.embedded = False

        self.roi_summary = self.roi_choice = self.roi_frame = self.current_roi = None
        self.embedder = self.embed = self.embed_plot = None
        self.class_fname = self.class_line = self.class_num = self.class_tag = None
        self.binx = self.biny = self.connect_id = None

        self._init_ui()

    def _init_ui(self):
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
        button.clicked.connect(self._do_embedding)
        hbox.addWidget(button)
        self.track_flag = QtWidgets.QCheckBox('Draw ROI', self)
        self.track_flag.setChecked(False)
        self.track_flag.stateChanged.connect(self._track_flag_changed)
        hbox.addWidget(self.track_flag)
        hbox.addStretch(1)

        hbox = QtWidgets.QHBoxLayout()
        self.vbox.addLayout(hbox)
        label = QtWidgets.QLabel('X-axis:', self)
        hbox.addWidget(label)
        self.x_axis_num = QtWidgets.QComboBox(self)
        for i in range(4):
            self.x_axis_num.addItem(str(i))
        self.x_axis_num.currentIndexChanged.connect(self._gen_hist)
        hbox.addWidget(self.x_axis_num)
        label = QtWidgets.QLabel('Y-axis:', self)
        hbox.addWidget(label)
        self.y_axis_num = QtWidgets.QComboBox(self)
        for i in range(4):
            self.y_axis_num.addItem(str(i))
        self.y_axis_num.setCurrentIndex(1)
        self.y_axis_num.currentIndexChanged.connect(self._gen_hist)
        hbox.addWidget(self.y_axis_num)
        hbox.addStretch(1)

        self.vbox.addStretch(1)

    def _do_embedding(self):
        converted = self.conversion.converted
        if converted is None:
            #self.conversion.convert_frames()
            self.conversion.converted = np.load(self.parent.output_folder+'/converted.npy')
            converted = self.conversion.converted

        method_ind = self.method.currentIndex()
        print('Doing %s' % self.method.currentText())
        if method_ind == 0:
            self.embedder = manifold.SpectralEmbedding(n_components=4, n_jobs=-1)
        elif method_ind == 1:
            self.embedder = manifold.Isomap(n_components=4, n_jobs=-1)
        elif method_ind == 2:
            self.embedder = manifold.LocallyLinearEmbedding(n_components=4,
                                                            n_jobs=-1,
                                                            n_neighbors=20,
                                                            method='modified')
        elif method_ind == 3:
            self.embedder = manifold.LocallyLinearEmbedding(n_components=4,
                                                            n_jobs=-1,
                                                            n_neighbors=20,
                                                            method='hessian',
                                                            eigen_solver='dense')
        elif method_ind == 4:
            self.embedder = manifold.MDS(n_components=4, n_jobs=-1)
        elif method_ind == 5:
            self.embedder = manifold.TSNE(n_components=3, init='pca')
        self.embedder.fit(converted)
        self.embed = self.embedder.embedding_
        self.embed_plot = self.embed

        self._gen_hist()
        self._plot_embedding()
        if not self.embedded:
            self._add_classes_frame()
        self.embedded = True

    def _plot_embedding(self):
        try:
            for point in self.roi_list:
                point.remove()
        except (ValueError, AttributeError):
            pass

        fig = self.frame.fig
        fig.clear()
        subp = fig.add_subplot(111)
        eplot = self.embed_plot
        xnum = int(self.x_axis_num.currentText())
        ynum = int(self.y_axis_num.currentText())
        subp.hist2d(eplot[:, xnum], eplot[:, ynum], bins=[self.binx, self.biny],
                    vmax=float(self.frame.rangestr.text()), cmap=self.parent.cmap)
        subp.set_title(self.method.currentText())
        for patch in self.roi_list:
            patch.set_transform(subp.transData)
            subp.add_patch(patch)
        fig.add_subplot(subp)
        self.frame.canvas.draw()

    def _gen_hist(self):
        xnum = int(self.x_axis_num.currentText())
        ynum = int(self.y_axis_num.currentText())
        _, self.binx, self.biny = np.histogram2d(self.embed[:, xnum],
                                                 self.embed[:, ynum],
                                                 bins=100)

        delx = self.binx[1] - self.binx[0]
        dely = self.biny[1] - self.biny[0]
        self.binx = np.insert(self.binx, 0, [self.binx[0]-6*delx, self.binx[0]-delx])
        self.binx = np.insert(self.binx, len(self.binx), [self.binx[-1]+delx, self.binx[-1]+6*delx])
        self.biny = np.insert(self.biny, 0, [self.biny[0]-6*dely, self.biny[0]-dely])
        self.biny = np.insert(self.biny, len(self.biny), [self.biny[-1]+dely, self.biny[-1]+6*dely])

    def _add_classes_frame(self):
        self.vbox.setStretch(self.vbox.count()-1, 0)

        hbox = QtWidgets.QHBoxLayout()
        self.vbox.addLayout(hbox)
        self.class_line = QtWidgets.QGridLayout()
        hbox.addLayout(self.class_line)
        hbox.addStretch(1)
        self.class_num = QtWidgets.QButtonGroup()
        self._refresh_classes()

        hbox = QtWidgets.QHBoxLayout()
        self.vbox.addLayout(hbox)
        button = QtWidgets.QPushButton('Show', self)
        button.clicked.connect(self._show_selected_class)
        hbox.addWidget(button)
        button = QtWidgets.QPushButton('See all', self)
        button.clicked.connect(self._show_all_classes)
        hbox.addWidget(button)
        button = QtWidgets.QPushButton('Refresh', self)
        button.clicked.connect(self._refresh_classes)
        hbox.addWidget(button)
        hbox.addStretch(1)

        self.vbox.addStretch(1)

    def _show_selected_class(self):
        ind = self.conversion.indices
        key_pos = np.where(self.classes.key_pos[ind] == self.class_num.checkedId())[0]
        self.embed_plot = self.embed[key_pos]
        self._plot_embedding()

    def _show_all_classes(self):
        self.embed_plot = self.embed
        self._plot_embedding()

    def _refresh_classes(self):
        for i in reversed(range(self.class_line.count())):
            widg = self.class_line.itemAt(i).widget()
            self.class_line.removeWidget(widg)
            widg.setParent(None)
        for i, key in enumerate(self.classes.key):
            if key == ' ':
                text = '  '
            else:
                text = key
            button = QtWidgets.QRadioButton(text, self)
            button.clicked.connect(self._show_selected_class)
            if i == 0:
                button.setChecked(True)
            self.class_num.addButton(button, i)
            self.class_line.addWidget(button, i/5, i%5)

    def _track_flag_changed(self):
        if self.track_flag.isChecked():
            if self.embed is not None:
                self.connect_id = self.frame.canvas.mpl_connect('button_press_event',
                                                                self._track_positions)
        else:
            self._end_track_positions()

    def _track_positions(self, event=None):
        x = event.xdata
        y = event.ydata
        self.click_points_list.append(
            self.frame.fig.get_axes()[0].plot(
                [x], [y], marker='.', markersize=8., color='white')[0])
        self.frame.canvas.draw()
        self.positions.append([x, y])

    def _end_track_positions(self):
        pos = np.array(self.positions)
        if pos.size == 0:
            return
        xnum = int(self.x_axis_num.currentText())
        ynum = int(self.y_axis_num.currentText())
        pos = np.append(pos, pos[-1]).reshape(-1,2)
        self.path_list.append(matplotlib.path.Path(pos, closed=True))
        points_inside = np.array([self.path_list[-1].contains_point((point[xnum], point[ynum]))
                                  for point in self.embed])
        sys.stderr.write('%d/%d frames inside ROI %d\n' % (points_inside.sum(),
                                                           len(points_inside),
                                                           len(self.points_inside_list)))
        self.points_inside_list.append(self.conversion.indices[np.where(points_inside)[0]])

        self.roi_list.append(
            matplotlib.patches.PathPatch(
                self.path_list[-1],
                color='white',
                fill=False,
                linewidth=2.,
                figure=self.frame.fig,
            )
        )
        self.frame.fig.get_axes()[0].add_artist(self.roi_list[-1])
        for point in self.click_points_list:
            point.remove()
        self.frame.canvas.draw()

        self.frame.canvas.mpl_disconnect(self.connect_id)
        self.positions = []
        self.click_points_list = []
        if self.roi_summary is None:
            self._add_roi_frame()
        elif self.roi_summary.text() == '':
            self.roi_frame.show()
        self._gen_roi_summary()
        self._add_roi_radiobutton(len(self.roi_list)-1)

    def _add_roi_frame(self):
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
        button.clicked.connect(self._clear_roi)
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
        gui_utils.add_scroll_hbox(self, hbox)
        hbox.addStretch(1)

        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        self.class_tag = QtWidgets.QLineEdit('', self)
        self.class_tag.setFixedWidth(24)
        hbox.addWidget(self.class_tag)
        button = QtWidgets.QPushButton('Apply Class', self)
        button.clicked.connect(self._apply_class)
        hbox.addWidget(button)
        hbox.addStretch(1)

        gui_utils.add_class_hbox(self, vbox)

    def _add_roi_radiobutton(self, num):
        button = QtWidgets.QRadioButton(str(num))
        if num == 0:
            button.setChecked(True)
        self.current_roi.addButton(button, num)
        self.roi_choice.addWidget(button, num/5, num%5)

    def _update_name(self):
        self.classes.fname = str(self.class_fname.text())

    def _gen_roi_summary(self):
        summary = 'Embedded frames = %d\n' % len(self.embed)
        for i, point in enumerate(self.points_inside_list):
            summary += '%3d:%-5d ' % (i, len(point))
            if i%5 == 4:
                summary += '\n'
        self.roi_summary.setText(summary)

    def _clear_roi(self):
        self._plot_embedding()
        for point in self.roi_list:
            point.remove()
        self.frame.canvas.draw()
        self.roi_list = []
        self.path_list = []
        self.points_inside_list = []
        for i in reversed(range(self.roi_choice.count())):
            widg = self.roi_choice.itemAt(i).widget()
            self.roi_choice.removeWidget(widg)
            widg.setParent(None)
        self.roi_frame.hide()
        self.roi_summary.setText('')

    def _prev_frame(self):
        num = int(self.frame.numstr.text())
        points = self.points_inside_list[self.current_roi.checkedId()]
        index = np.searchsorted(points, num, side='left') - 1
        if index < 0:
            index = 0
        self.frame.numstr.setText(str(points[index]))
        self.frame.plot_frame()

    def _next_frame(self):
        num = int(self.frame.numstr.text())
        points = self.points_inside_list[self.current_roi.checkedId()]
        index = np.searchsorted(points, num, side='left') + 1
        if index > len(points) - 1:
            index = len(points) - 1
        self.frame.numstr.setText(str(points[index]))
        self.frame.plot_frame()

    def _rand_frame(self):
        points = self.points_inside_list[self.current_roi.checkedId()]
        self.frame.numstr.setText(str(points[np.random.randint(len(points))]))
        self.frame.plot_frame()

    def _apply_class(self):
        roi_num = self.current_roi.checkedId()
        class_char = str(self.class_tag.text())
        self.classes.clist[self.points_inside_list[roi_num]] = class_char
        if self.class_line.count() != len(self.classes.key) + 1:
            self._refresh_classes()
        self.classes.gen_summary()
        self.classes.unsaved = True
