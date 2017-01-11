import numpy as np
import sys
import os
import string
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import Tkinter as Tk
import ttk

class Frame_panel(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        ttk.Frame.__init__(self, parent.master, *args, **kwargs)
        
        self.parent = parent
        self.emc_reader = self.parent.emc_reader
        self.num_frames = self.parent.num_frames
        self.cmap = self.parent.cmap
        self.mode = None
        
        self.numstr = Tk.StringVar(); self.numstr.set(str(0))
        self.rangestr = Tk.StringVar(); self.rangestr.set(str(10))
        
        self.init_UI()

    def init_UI(self):
        self.fig = plt.figure(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas.show()
        self.canvas_widget.pack(fill='both', expand=1)
        
        self.options = ttk.Frame(self.parent.master, relief=Tk.GROOVE, borderwidth=5, width=400, height=200)
        self.options.grid(row=1, column=0, sticky='nsew')
        
        line = ttk.Frame(self.options)
        line.pack(fill=Tk.X)
        ttk.Label(line, text='Frame number: ').pack(side=Tk.LEFT)
        ttk.Entry(line, textvariable=self.numstr, width=8).pack(side=Tk.LEFT)
        ttk.Label(line, text='/%d'%self.num_frames).pack(side=Tk.LEFT)
        ttk.Entry(line, textvariable=self.rangestr, width=6).pack(side=Tk.RIGHT)
        ttk.Label(line, text='PlotMax: ').pack(side=Tk.RIGHT)
        
        line = ttk.Frame(self.options)
        line.pack(fill=Tk.X)
        ttk.Button(line, text='Plot', command=self.plot_frame).pack(side=Tk.LEFT)
        ttk.Button(line, text='Prev', command=self.prev_frame).pack(side=Tk.LEFT)
        ttk.Button(line, text='Next', command=self.next_frame).pack(side=Tk.LEFT)
        ttk.Button(line, text='Random', command=self.rand_frame).pack(side=Tk.LEFT)
        ttk.Button(line, text='Quit', command=self.parent.quit).pack(side=Tk.RIGHT)
        
        self.master.bind('<Return>', self.plot_frame)
        self.master.bind('<KP_Enter>', self.plot_frame)
        self.master.bind('<Control-n>', self.next_frame)
        self.master.bind('<Control-p>', self.prev_frame)
        self.master.bind('<Control-r>', self.rand_frame)
        self.master.bind('<Control-q>', self.parent.quit)
        self.canvas_widget.bind('<Button-1>', self.frame_focus)
        self.canvas_widget.bind('<Right>', self.next_frame)
        self.canvas_widget.bind('<Left>', self.prev_frame)
        self.canvas_widget.bind('<Up>', self.next_frame)
        self.canvas_widget.bind('<Down>', self.prev_frame)
        
        self.plot_frame()

    def plot_frame(self, event=None, embed=None, force_frame=False):
        if self.mode is not None:
            mode = self.mode.get()
        else:
            mode = 0
        
        try:
            num = int(self.numstr.get())
        except ValueError:
            print 'Frame number must be integer'
            return
        
        if num < 0 or num >= self.num_frames:
            sys.stderr.write('Frame number %d out of range!\n' % num)
            return
        
        frame = self.emc_reader.get_frame(num)
        
        if mode == 2:
            # Conversion panel
            s = plt.subplot(121)
            s.imshow(frame, vmin=0, vmax=float(self.rangestr.get()), interpolation='none', cmap=self.cmap)
            s.set_title("%d photons" % frame.sum())
            self.fig.add_subplot(s)
            
            s = plt.subplot(122)
            pframe = self.parent.conversion_panel.polar.convert(frame)
            s.imshow(pframe, vmin=0, vmax=float(self.rangestr.get()), interpolation='none', cmap=self.cmap, aspect=float(pframe.shape[1])/pframe.shape[0])
            title = 'Polar representation'
            self.fig.add_subplot(s)
        elif (not force_frame) and mode == 3 and self.parent.embedding_panel.embed is not None:
            # Embedding 
            ep = self.parent.embedding_panel
            for p in ep.roi_list:
                self.canvas_widget.tag_raise(p)
            for p in ep.click_points_list:
                self.canvas_widget.tag_raise(p)
            
            plt.gcf().clear()
            s = plt.subplot(111)
            e = ep.embed_plot
            try:
                xnum = int(ep.x_axis_num.get())
                ynum = int(ep.y_axis_num.get())
            except ValueError:
                print 'Need axes numbers to be integers'
                return
            s.hist2d(e[:,xnum], e[:,ynum], bins=[ep.binx, ep.biny], vmax=float(self.rangestr.get()))
            title = 'Spectral embedding'
            self.fig.add_subplot(s)
        else:
            if mode == 3:
                ep = self.parent.embedding_panel
                for p in ep.roi_list:
                    self.canvas_widget.tag_lower(p)
                for p in ep.click_points_list:
                    self.canvas_widget.tag_lower(p)
            
            plt.gcf().clear()
            s = plt.subplot(111)
            s.imshow(frame, vmin=0, vmax=float(self.rangestr.get()), interpolation='none', cmap=self.cmap)
            title = '%d photons' % frame.sum()
            if mode == 1:
                title += ' (%s)' % self.parent.classes.clist[num]
            if mode == 4 and self.parent.mlp_panel.predictions is not None:
                title += ' [%s]' % self.parent.mlp_panel.predictions[num]
            if self.mode is None and self.parent.blacklist is not None and self.parent.blacklist[num] == 1:
                title += ' (bad frame)'
            s.set_title(title)
            self.fig.add_subplot(s)
        self.canvas.show()

    def next_frame(self, event=None):
        num = int(self.numstr.get()) + 1
        if num < self.num_frames:
            self.numstr.set(str(num))
            self.plot_frame()

    def prev_frame(self, event=None):
        num = int(self.numstr.get()) - 1
        if num > -1:
            self.numstr.set(str(num))
            self.plot_frame()

    def rand_frame(self, event=None):
        num = np.random.randint(0, self.num_frames)
        self.numstr.set(str(num))
        self.plot_frame()

    def frame_focus(self, event=None):
        self.canvas_widget.focus_set()

