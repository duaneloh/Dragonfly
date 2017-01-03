import numpy as np
import sys
import os
import string
import Tkinter as Tk
import ttk
import matplotlib.path
from sklearn import manifold

class Embedding_panel(ttk.Frame, object):
    def __init__(self, parent, *args, **kwargs):
        ttk.Frame.__init__(self, parent.master, *args, **kwargs)
        self.parent = parent
        self.classes = self.parent.classes
        self.conversion = self.parent.conversion_panel
        self.manual = self.parent.manual_panel
        
        self.track_flag = Tk.IntVar(); self.track_flag.set(0)
        self.current_roi = Tk.IntVar(); self.current_roi.set(0)
        self.class_tag = Tk.StringVar(); self.class_tag.set('')
        self.class_num = Tk.IntVar(); self.class_num.set(0)
        self.x_axis_num = Tk.StringVar(); self.x_axis_num.set('0')
        self.y_axis_num = Tk.StringVar(); self.y_axis_num.set('1')
        self.positions = []
        self.poly_positions = []
        self.roi_list = []
        self.path_list = []
        self.points_inside_list = []
        self.click_points_list = []
        self.embed = None
        self.roi_summary = None
        self.embedded = False
        
        self.init_UI()

    def init_UI(self):
        line = ttk.Frame(self); line.pack(fill=Tk.X)
        ttk.Label(line, text='Spectral manifold embedding').pack(side=Tk.LEFT, fill=Tk.X)
        
        line = ttk.Frame(self); line.pack(fill=Tk.X)
        ttk.Label(line, text='Frame range:').pack(side=Tk.LEFT)
        ttk.Entry(line, textvariable=self.conversion.first_frame, width=8).pack(side=Tk.LEFT)
        ttk.Label(line, text='-').pack(side=Tk.LEFT)
        ttk.Entry(line, textvariable=self.conversion.last_frame, width=8).pack(side=Tk.LEFT)
        
        line = ttk.Frame(self); line.pack(fill=Tk.X)
        ttk.Button(line, text='Embed', command=self.do_embedding).pack(side=Tk.LEFT)
        ttk.Checkbutton(line, text='Draw ROI', variable=self.track_flag, command=self.track_flag_changed).pack(side=Tk.LEFT)
        
        line = ttk.Frame(self); line.pack(fill=Tk.X)
        ttk.Label(line, text='X-axis:').pack(side=Tk.LEFT)
        entry = ttk.Entry(line, textvariable=self.x_axis_num, width=2)
        entry.pack(side=Tk.LEFT)
        entry.bind('<Return>', self.gen_hist)
        ttk.Label(line, text='Y-axis:').pack(side=Tk.LEFT)
        entry = ttk.Entry(line, textvariable=self.y_axis_num, width=2)
        entry.pack(side=Tk.LEFT)
        entry.bind('<Return>', self.gen_hist)

    def do_embedding(self, event=None):
        ang_corr = self.parent.ang_corr
        if ang_corr is None:
            #self.conversion.convert_frames()
            self.parent.ang_corr = np.load(self.parent.output_folder+'/ang_corr.npy') #FIXME For debugging
            ang_corr = self.parent.ang_corr
        
        self.spectral = manifold.SpectralEmbedding(n_components=4, n_jobs=-1)
        self.spectral.fit(ang_corr)
        self.embed = self.spectral.embedding_
        self.embed_plot = self.embed
        
        self.gen_hist()
        self.parent.plot_frame()
        if not self.embedded:
            self.add_classes_frame()
        self.embedded = True

    def gen_hist(self, event=None):
        try:
            xnum = int(self.x_axis_num.get())
            ynum = int(self.y_axis_num.get())
        except ValueError:
            print 'Need axes numbers to be integers'
            return
        self.hist2d, self.binx, self.biny = np.histogram2d(self.embed[:,xnum], self.embed[:,ynum], bins=100)
        
        delx = self.binx[1] - self.binx[0]
        dely = self.biny[1] - self.biny[0]
        self.binx = np.insert(self.binx, 0, [self.binx[0]-6*delx, self.binx[0]-delx])
        self.binx = np.insert(self.binx, len(self.binx), [self.binx[-1]+delx, self.binx[-1]+6*delx])
        self.biny = np.insert(self.biny, 0, [self.biny[0]-6*dely, self.biny[0]-dely])
        self.biny = np.insert(self.biny, len(self.biny), [self.biny[-1]+dely, self.biny[-1]+6*dely])

    def add_classes_frame(self):
        self.classes_frame = ttk.Frame(self, borderwidth=4, relief='groove'); self.classes_frame.pack(fill=Tk.X)
        
        self.classes_line = ttk.Frame(self.classes_frame); self.classes_line.pack(fill=Tk.X)
        for i, k in enumerate(self.classes.key):
            ttk.Radiobutton(self.classes_line, text=k, variable=self.class_num, value=i).grid(row=i/5,column=i%5)
        
        line = ttk.Frame(self.classes_frame); line.pack(fill=Tk.X)
        ttk.Button(line, text='Show', command=self.show_selected_class).pack(side=Tk.LEFT)
        ttk.Button(line, text='See all', command=self.show_all_classes).pack(side=Tk.LEFT)
        ttk.Button(line, text='Refresh', command=self.refresh_classes).pack(side=Tk.LEFT)

    def show_selected_class(self, event=None):
        first = int(self.conversion.first_frame.get())
        last = int(self.conversion.last_frame.get())
        key_pos = np.where(self.classes.key_pos == self.class_num.get())[0]
        key_pos = key_pos[(key_pos>=first) & (key_pos<last)] - first
        self.embed_plot = self.embed[key_pos]
        self.parent.plot_frame()

    def show_all_classes(self, event=None):
        self.embed_plot = self.embed
        self.parent.plot_frame()

    def refresh_classes(self,event=None):
        for c in self.classes_line.winfo_children():
            c.destroy()
        for i, k in enumerate(self.classes.key):
            ttk.Radiobutton(self.classes_line, text=k, variable=self.class_num, value=i).grid(row=i/5,column=i%5)

    def track_flag_changed(self, event=None):
        if self.track_flag.get() == 1:
            if self.embed is not None:
                self.connect_id = self.parent.canvas.mpl_connect('button_press_event', self.track_positions)
        else:
            self.end_track_positions()

    def track_positions(self, event=None):
        self.event = event
        self.click_points_list.append(self.parent.canvas_widget.create_line([event.guiEvent.x, event.guiEvent.y, event.guiEvent.x+1, event.guiEvent.y+1], fill='white', width=4.0))
        self.positions.append([event.xdata, event.ydata])
        self.poly_positions.append([event.guiEvent.x, event.guiEvent.y])

    def end_track_positions(self):
        pos = np.array(self.positions)
        if pos.size == 0:
            return
        try:
            xnum = int(self.x_axis_num.get())
            ynum = int(self.y_axis_num.get())
        except ValueError:
            print 'Need axes numbers to be integers'
            return
        pos = np.append(pos, pos[-1]).reshape(-1,2)
        self.path_list.append(matplotlib.path.Path(pos, closed=True))
        points_inside = np.array([self.path_list[-1].contains_point((p[xnum], p[ynum])) for p in self.embed])
        print '%d/%d frames inside ROI %d' % (points_inside.sum(), len(points_inside), len(self.points_inside_list))
        self.points_inside_list.append(np.where(points_inside)[0] + int(self.conversion.first_frame.get()))
        
        self.roi_list.append(
            self.parent.canvas_widget.create_polygon(
                np.array(self.poly_positions).flatten().tolist(), 
                fill='',
                outline='white',
                width=2
            )
        )
        
        self.parent.canvas.mpl_disconnect(self.connect_id)
        self.positions = []
        self.poly_positions = []
        if self.roi_summary is None:
            self.add_roi_frame()
        self.gen_roi_summary()
        self.add_roi_radiobutton(len(self.roi_list)-1)

    def add_roi_frame(self):
        self.roi_summary = Tk.StringVar(); self.roi_summary.set('')
        self.roi_frame = ttk.Frame(self, borderwidth=4, relief='groove'); self.roi_frame.pack(fill=Tk.X)
        
        line = ttk.Frame(self.roi_frame); line.pack(fill=Tk.X)
        ttk.Label(line, textvariable=self.roi_summary).pack(side=Tk.LEFT)
        
        line = ttk.Frame(self.roi_frame); line.pack(fill=Tk.X)
        ttk.Button(line, text='Clear ROIs', command=self.clear_roi).pack(side=Tk.LEFT)
        
        self.roi_choice = ttk.Frame(self.roi_frame); self.roi_choice.pack(fill=Tk.X)
        
        line = ttk.Frame(self.roi_frame); line.pack(fill=Tk.X)
        ttk.Button(line, text='Prev', command=self.prev_frame).pack(side=Tk.LEFT)
        ttk.Button(line, text='Next', command=self.next_frame).pack(side=Tk.LEFT)
        ttk.Button(line, text='Random', command=self.random_frame).pack(side=Tk.LEFT)
        
        line = ttk.Frame(self.roi_frame); line.pack(fill=Tk.X)
        ttk.Entry(line, textvariable=self.class_tag, width=2).pack(side=Tk.LEFT)
        ttk.Button(line, text='Apply Class', command=self.apply_class).pack(side=Tk.LEFT)
        
        line = ttk.Frame(self.roi_frame); line.pack(fill=Tk.X)
        ttk.Entry(line, textvariable=self.manual.class_list_fname).pack(side=Tk.LEFT)
        ttk.Button(line, text='Save Classes', command=self.manual.save_class_list).pack(side=Tk.LEFT)

    def add_roi_radiobutton(self, num):
        ttk.Radiobutton(self.roi_choice, text=str(num), variable=self.current_roi, value=num).grid(row=num/5, column=num%5)

    def gen_roi_summary(self):
        summary = 'Embedded frames = %d\n' % len(self.embed)
        for i, p in enumerate(self.points_inside_list):
            summary += '%3d:%-5d ' % (i, len(p))
            if i%5 == 4:
                summary += '\n'
        self.roi_summary.set(summary)

    def clear_roi(self):
        self.roi_summary.set('')
        for p in self.roi_list:
            self.parent.canvas_widget.delete(p)
        for p in self.click_points_list:
            self.parent.canvas_widget.delete(p)
        self.roi_list = []
        self.click_points_list = []
        self.path_list = []
        self.points_inside_list = []
        self.roi_frame.pack_forget()
        self.roi_frame.destroy()
        self.roi_summary = None

    def prev_frame(self, event=None):
        num = int(self.parent.numstr.get())
        points = self.points_inside_list[self.current_roi.get()]
        index = np.searchsorted(points, num, side='left') - 1
        if index < 0:
            index = 0
        self.parent.numstr.set(str(points[index]))
        self.parent.plot_frame(force_frame=True)

    def next_frame(self, event=None):
        num = int(self.parent.numstr.get())
        points = self.points_inside_list[self.current_roi.get()]
        index = np.searchsorted(points, num, side='left') + 1
        if index > len(points) - 1:
            index = len(points) - 1
        self.parent.numstr.set(str(points[index]))
        self.parent.plot_frame(force_frame=True)

    def random_frame(self, event=None):
        points = self.points_inside_list[self.current_roi.get()]
        self.parent.numstr.set(str(points[np.random.randint(len(points))]))
        self.parent.plot_frame(force_frame=True)

    def apply_class(self, event=None):
        roi_num = self.current_roi.get()
        class_char = self.class_tag.get()
        self.classes.clist[self.points_inside_list[roi_num]] = class_char
        self.classes.gen_summary()
        self.classes.unsaved = True

    def grid_forget(self):
        for p in self.roi_list:
            self.parent.canvas_widget.tag_lower(p)
        for p in self.click_points_list:
            self.parent.canvas_widget.tag_lower(p)
        super(Embedding_panel, self).grid_forget()
