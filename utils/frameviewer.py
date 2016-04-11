#!/usr/bin/env python

import numpy as np
import sys
import os
import ConfigParser
from py_src import py_utils
from py_src import read_config

import matplotlib.pyplot as plt
import Tkinter as Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Frameviewer():
    def __init__(self, master, photons_list, frame_shape, cmap='jet', mask=None):
        self.master = master
        self.photons_list = photons_list
        self.num_files = len(photons_list)
        self.frame_shape = frame_shape
        self.cmap = cmap
        if mask == None:
            self.mask = np.ones(self.frame_shape)
        else:
            self.mask = np.fromfile(mask, '=u1').reshape(self.frame_shape)
            self.mask[self.mask==0] = 1
            self.mask[self.mask==2] = 0
        
        self.numstr = Tk.StringVar(); self.numstr.set(str(0))
        self.rangestr = Tk.StringVar(); self.rangestr.set(str(10))
        
        self.parse_headers()
        self.init_geom()
        self.init_UI()

    def init_UI(self):
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)
        
        fig_frame = Tk.Frame(self.master)
        fig_frame.grid(row=0, column=0, sticky='nsew')
        fig_frame.columnconfigure(0, weight=1)
        fig_frame.rowconfigure(0, weight=1)
        
        self.fig = plt.figure(figsize=(6, 6))
        #self.fig.subplots_adjust(left=0.0, bottom=0.00, right=0.99, wspace=0.0)
        self.canvas = FigureCanvasTkAgg(self.fig, fig_frame)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(fill='both', expand=1)
        
        self.options = Tk.Frame(self.master, relief=Tk.GROOVE, borderwidth=5, width=400, height=200)
        self.options.grid(row=1, column=0, sticky='nsew')
        
        line = Tk.Frame(self.options)
        line.pack(fill=Tk.X)
        Tk.Label(line, text='Frame number: ').pack(side=Tk.LEFT)
        Tk.Entry(line, textvariable=self.numstr, width=10).pack(side=Tk.LEFT, fill=Tk.X, expand=1)
        Tk.Label(line, text='/%d'%self.num_frames).pack(side=Tk.LEFT)
        Tk.Label(line, text='PlotMax: ').pack(side=Tk.LEFT, fill=Tk.X)
        Tk.Entry(line, textvariable=self.rangestr, width=6).pack(side=Tk.LEFT)
        
        line = Tk.Frame(self.options)
        line.pack(fill=Tk.X)
        Tk.Button(line, text='Plot', command=self.plot_frame).pack(side=Tk.LEFT)
        Tk.Button(line, text='Prev', command=self.prev_frame).pack(side=Tk.LEFT)
        Tk.Button(line, text='Next', command=self.next_frame).pack(side=Tk.LEFT)
        Tk.Button(line, text='Random', command=self.rand_frame).pack(side=Tk.LEFT)
        Tk.Button(line, text='Quit', command=self.quit).pack(side=Tk.RIGHT)
        
        self.master.bind('<Return>', self.plot_frame)
        self.master.bind('<KP_Enter>', self.plot_frame)
        self.master.bind('<Control-n>', self.next_frame)
        self.master.bind('<Control-p>', self.prev_frame)
        self.master.bind('<Control-r>', self.rand_frame)
        self.master.bind('<Control-q>', self.quit)
        self.master.bind('<Up>', self.next_frame)
        self.master.bind('<Down>', self.prev_frame)
        
        self.plot_frame()

    def init_geom(self):
        self.x, self.y = np.indices(self.frame_shape)
        self.x = self.x.flatten()
        self.y = self.y.flatten()

    def parse_headers(self):
        self.num_data_list = []
        self.ones_accum_list = []
        self.multi_accum_list = []
        
        # For each emc file, read num_data and generate ones_accum and multi_accum
        for photons_file in self.photons_list:
            # Read photon data
            with open(photons_file, 'rb') as f:
                num_data = np.fromfile(f, dtype='i4', count=1)[0]
                f.seek(1024, 0)
                ones = np.fromfile(f, dtype='i4', count=num_data)
                multi = np.fromfile(f, dtype='i4', count=num_data)
            self.num_data_list.append(num_data)
            self.ones_accum_list.append(np.cumsum(ones))
            self.multi_accum_list.append(np.cumsum(multi))
        
        self.num_data_list = np.cumsum(self.num_data_list)
        self.num_frames = self.num_data_list[-1]

    def read_frame(self, file_num, frame_num):
        with open(self.photons_list[file_num], 'rb') as f:
            num_data = np.fromfile(f, dtype='i4', count=1)[0]
            
            ones_accum = self.ones_accum_list[file_num]
            multi_accum = self.multi_accum_list[file_num]
            
            if frame_num == 0:
                ones_offset = 0
                multi_offset = 0
                ones_size = ones_accum[frame_num]
                multi_size = multi_accum[frame_num]
            else:
                ones_offset = ones_accum[frame_num - 1]
                multi_offset = multi_accum[frame_num - 1]
                ones_size = ones_accum[frame_num] - ones_accum[frame_num - 1]
                multi_size = multi_accum[frame_num] - multi_accum[frame_num - 1]
            
            f.seek(1024 + num_data*8 + ones_offset*4, 0)
            place_ones = np.fromfile(f, dtype='i4', count=ones_size)
            f.seek(1024 + num_data*8 + ones_accum[-1]*4 + multi_offset*4, 0)
            place_multi = np.fromfile(f, dtype='i4', count=multi_size)
            f.seek(1024 + num_data*8 + ones_accum[-1]*4 + multi_accum[-1]*4 + multi_offset*4, 0)
            count_multi = np.fromfile(f, dtype='i4', count=multi_size)
        
        frame = np.zeros(self.frame_shape, dtype='i4')
        np.add.at(frame, (self.x[place_ones], self.y[place_ones]), 1)
        np.add.at(frame, (self.x[place_multi], self.y[place_multi]), count_multi)
        
        return frame * self.mask

    def plot_frame(self, event=None):
        num = int(self.numstr.get())
        if num < 0 or num >= self.num_frames:
            sys.stderr.write('Frame number %d out of range!\n')
            return
        
        file_num = np.where(num < self.num_data_list)[0][0]
        if file_num == 0:
            frame_num = num
        else:
            frame_num = num - self.num_data_list[file_num-1]
        frame = self.read_frame(file_num, frame_num)
        
        s = plt.subplot(111)
        s.imshow(frame, vmin=0, vmax=float(self.rangestr.get()), interpolation='none', cmap=self.cmap)
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
    
    def quit(self, event=None):
        self.master.quit()

if __name__ == '__main__':
    parser = py_utils.my_argparser(description='Utility for viewing frames of the emc file (list)')
    parser.add_argument('--cmap', help='Matplotlib color map (default: jet)')
    parser.add_argument('--mask', help='Name of mask file of type uint8 (default: None)')
    args = parser.special_parse_args()
    
    try:
        pfile = read_config.get_filename(args.config_file, 'emc', 'in_photons_file')
        print 'Using in_photons_file: %s' % pfile
        photons_list = [pfile]
    except ConfigParser.NoOptionError:
        plist = read_config.get_filename(args.config_file, 'emc', 'in_photons_list')
        print 'Using in_photons_list: %s' % plist
        with open(plist, 'r') as f:
            photons_list = map(lambda x: x.rstrip(), f.readlines())
    
    pm = read_config.get_detector_config(args.config_file, show=args.vb)
    
    root = Tk.Tk()
    Frameviewer(root, photons_list, (pm['dets_x'], pm['dets_y']), cmap=args.cmap, mask=args.mask)
    root.mainloop()
