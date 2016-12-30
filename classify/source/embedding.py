import numpy as np
import sys
import os
import string
import Tkinter as Tk
import ttk
import matplotlib.path
from sklearn import manifold

class Embedding_panel(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        ttk.Frame.__init__(self, parent.master, *args, **kwargs)
        self.parent = parent
        self.classes = self.parent.classes
        
        self.track_flag = Tk.IntVar(); self.track_flag.set(0)
        self.current_roi = Tk.IntVar(); self.current_roi.set(0)
        self.class_tag = Tk.StringVar(); self.class_tag.set('')
        self.positions = []
        self.poly_positions = []
        self.roi_list = []
        self.path_list = []
        self.points_inside_list = []
        self.click_points_list = []
        self.embed = None
        self.roi_summary = None
        
        self.init_UI()

    def init_UI(self):
        line = ttk.Frame(self); line.pack(fill=Tk.X)
        ttk.Label(line, text='Spectral manifold embedding').pack(side=Tk.LEFT, fill=Tk.X)
        
        line = ttk.Frame(self); line.pack(fill=Tk.X)
        ttk.Label(line, text='Frame range:').pack(side=Tk.LEFT)
        ttk.Entry(line, textvariable=self.parent.conversion_panel.first_frame, width=8).pack(side=Tk.LEFT)
        ttk.Label(line, text='-').pack(side=Tk.LEFT)
        ttk.Entry(line, textvariable=self.parent.conversion_panel.last_frame, width=8).pack(side=Tk.LEFT)
        
        line = ttk.Frame(self); line.pack(fill=Tk.X)
        ttk.Button(line, text='Embed', command=self.do_embedding).pack(side=Tk.LEFT)
        ttk.Checkbutton(line, text='Draw ROI', variable=self.track_flag, command=self.track_flag_changed).pack(side=Tk.LEFT)

    def do_embedding(self, event=None):
        ang_corr = self.parent.ang_corr
        if ang_corr is None:
            #self.parent.conversion_panel.convert_frames()
            self.parent.ang_corr = np.load('data/ang_corr.npy') #FIXME For debugging
            ang_corr = self.parent.ang_corr
        self.spectral = manifold.SpectralEmbedding(n_components=4)
        self.spectral.fit(ang_corr.reshape(len(ang_corr), -1))
        self.embed = self.spectral.embedding_
        #self.hist2d, self.binx, self.biny = np.histogram2d(self.embed[:,0], self.embed[:,1], bins=100)
        self.parent.plot_frame()

    def track_flag_changed(self, event=None):
        if self.track_flag.get() == 1:
            if self.embed is not None:
                self.connect_id = self.parent.canvas.mpl_connect('button_press_event', self.track_positions)
        else:
            self.end_track_positions()

    def track_positions(self, event=None):
        #print 'Coords:', event.xdata, event.ydata
        self.event = event
        self.click_points_list.append(self.parent.canvas_widget.create_line([event.guiEvent.x, event.guiEvent.y, event.guiEvent.x+1, event.guiEvent.y+1], fill='yellow', width=4.0))
        self.positions.append([event.xdata, event.ydata])
        self.poly_positions.append([event.guiEvent.x, event.guiEvent.y])

    def end_track_positions(self):
        pos = np.array(self.positions)
        pos = np.append(pos, pos[-1]).reshape(-1,2)
        self.path_list.append(matplotlib.path.Path(pos, closed=True))
        points_inside = np.array([self.path_list[-1].contains_point((p[0], p[1])) for p in self.embed])
        print points_inside.sum(), 'frames inside ROI out of', len(points_inside)
        self.points_inside_list.append(np.where(points_inside)[0] + int(self.parent.conversion_panel.first_frame.get()))
        
        self.roi_list.append(
            self.parent.canvas_widget.create_polygon(
                np.array(self.poly_positions).flatten().tolist(), 
                fill='',
                outline='yellow',
                width=2
            )
        )
        
        self.parent.canvas.mpl_disconnect(self.connect_id)
        self.positions = []
        self.poly_positions = []
        if self.roi_summary is None:
            self.add_frame_with_roi()
        self.gen_roi_summary()
        self.add_roi_radiobutton(len(self.roi_list)-1)

    def add_frame_with_roi(self):
        self.roi_summary = Tk.StringVar(); self.roi_summary.set('')
        self.roi_frame = ttk.Frame(self); self.roi_frame.pack(fill=Tk.X)
        
        line = ttk.Frame(self.roi_frame); line.pack(fill=Tk.X)
        ttk.Label(line, textvariable=self.roi_summary).pack(side=Tk.LEFT)
        
        line = ttk.Frame(self.roi_frame); line.pack(fill=Tk.X)
        ttk.Button(line, text='Clear ROIs', command=self.clear_roi).pack(side=Tk.LEFT)
        
        self.roi_line = ttk.Frame(self.roi_frame); self.roi_line.pack(fill=Tk.X)
        
        line = ttk.Frame(self.roi_frame); line.pack(fill=Tk.X)
        ttk.Button(line, text='Prev', command=self.prev_frame).pack(side=Tk.LEFT)
        ttk.Button(line, text='Next', command=self.next_frame).pack(side=Tk.LEFT)
        ttk.Button(line, text='Random', command=self.random_frame).pack(side=Tk.LEFT)
        
        line = ttk.Frame(self.roi_frame); line.pack(fill=Tk.X)
        ttk.Entry(line, textvariable=self.class_tag, width=2).pack(side=Tk.LEFT)
        ttk.Button(line, text='Apply Class', command=self.apply_class).pack(side=Tk.LEFT)
        
        line = ttk.Frame(self.roi_frame); line.pack(fill=Tk.X)
        ttk.Entry(line, textvariable=self.parent.manual_panel.class_list_fname).pack(side=Tk.LEFT)
        ttk.Button(line, text='Save Classes', command=self.parent.manual_panel.save_class_list).pack(side=Tk.TOP, anchor=Tk.W)

    def add_roi_radiobutton(self, num):
        if num > 0 and num % 5 == 0:
            self.roi_line = ttk.Frame(self); self.roi_line.pack(fill=Tk.X)
        ttk.Radiobutton(self.roi_line, text=str(num), variable=self.current_roi, value=num).pack(side=Tk.LEFT)

    def gen_roi_summary(self):
        summary = 'Embedded frames = %d\n' % len(self.embed)
        for i, p in enumerate(self.points_inside_list):
            summary += '|%3d|%6d|\n' % (i, len(p))
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
        self.classes.unsaved = True

