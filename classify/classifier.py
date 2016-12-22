#!/usr/bin/env python

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import Tkinter as Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import ConfigParser
from source import manual
from source import conversion
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+'/utils/')
from py_src import py_utils
from py_src import read_config

class Classifier():
    #def __init__(self, master, photons_list, frame_shape, det_scale=(0,0), cmap='jet', mask=False, det_fname=None):
    def __init__(self, master, config_file, cmap='jet', mask=False):
        self.master = master
        self.cmap = cmap
        self.config_file = config_file
        
        self.class_list = None
        self.mode_val = Tk.IntVar(); self.mode_val.set(0)
        self.numstr = Tk.StringVar(); self.numstr.set(str(0))
        self.rangestr = Tk.StringVar(); self.rangestr.set(str(10))
        
        self.get_config_params()
        self.parse_headers()
        self.init_geom(mask)
        self.init_UI()

    def init_UI(self):
        self.master.title('Dragonfly Frame Classifier')
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)
        self.master.protocol('WM_DELETE_WINDOW', self.master.quit)
        
        fig_frame = Tk.Frame(self.master)
        fig_frame.grid(row=0, column=0, sticky='nsew')
        fig_frame.columnconfigure(0, weight=1)
        fig_frame.rowconfigure(0, weight=1)
        
        self.fig = plt.figure(figsize=(6, 6))
        #self.fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, wspace=0.0)
        self.canvas = FigureCanvasTkAgg(self.fig, fig_frame)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(fill='both', expand=1)
        
        self.options = Tk.Frame(self.master, relief=Tk.GROOVE, borderwidth=5, width=400, height=200)
        self.options.grid(row=1, column=0, sticky='nsew')
        
        line = Tk.Frame(self.options)
        line.pack(fill=Tk.X)
        Tk.Label(line, text='Frame number: ').pack(side=Tk.LEFT)
        Tk.Entry(line, textvariable=self.numstr, width=10).pack(side=Tk.LEFT)
        Tk.Label(line, text='/%d'%self.num_frames).pack(side=Tk.LEFT)
        Tk.Entry(line, textvariable=self.rangestr, width=6).pack(side=Tk.RIGHT)
        Tk.Label(line, text='PlotMax: ').pack(side=Tk.RIGHT)
        
        line = Tk.Frame(self.options)
        line.pack(fill=Tk.X)
        Tk.Button(line, text='Plot', command=self.plot_frame).pack(side=Tk.LEFT)
        Tk.Button(line, text='Prev', command=self.prev_frame).pack(side=Tk.LEFT)
        Tk.Button(line, text='Next', command=self.next_frame).pack(side=Tk.LEFT)
        Tk.Button(line, text='Random', command=self.rand_frame).pack(side=Tk.LEFT)
        Tk.Button(line, text='Quit', command=self.quit).pack(side=Tk.RIGHT)
        
        menubar = Tk.Menu(self.master)
        modemenu = Tk.Menu(menubar, tearoff=0)
        modemenu.add_radiobutton(label='Display', underline=0, variable=self.mode_val, value=0, command=self.switch_mode)
        modemenu.add_radiobutton(label='Manual', underline=0, variable=self.mode_val, value=1, command=self.switch_mode)
        modemenu.add_radiobutton(label='Convert', underline=0, variable=self.mode_val, value=2, command=self.switch_mode)
        menubar.add_cascade(label='Mode', menu=modemenu, underline=0)
        self.master.config(menu=menubar)
        self.manual_panel = None
        self.conversion_panel = None
        
        self.master.bind('<Return>', self.plot_frame)
        self.master.bind('<KP_Enter>', self.plot_frame)
        self.master.bind('<Control-n>', self.next_frame)
        self.master.bind('<Control-p>', self.prev_frame)
        self.master.bind('<Control-r>', self.rand_frame)
        self.master.bind('<Control-q>', self.quit)
        self.canvas.get_tk_widget().bind('<Button-1>', self.frame_focus)
        self.canvas.get_tk_widget().bind('<Right>', self.next_frame)
        self.canvas.get_tk_widget().bind('<Left>', self.prev_frame)
        self.canvas.get_tk_widget().bind('<Up>', self.next_frame)
        self.canvas.get_tk_widget().bind('<Down>', self.prev_frame)
        
        self.plot_frame()

    def get_config_params(self):
        try:
            pfile = read_config.get_filename(self.config_file, 'emc', 'in_photons_file')
            print 'Using in_photons_file: %s' % pfile
            self.photons_list = [pfile]
        except ConfigParser.NoOptionError:
            plist = read_config.get_filename(self.config_file, 'emc', 'in_photons_list')
            print 'Using in_photons_list: %s' % plist
            with open(plist, 'r') as f:
                self.photons_list = map(lambda x: x.rstrip(), f.readlines())
        
        pm = read_config.get_detector_config(self.config_file)
        
        self.num_files = len(self.photons_list)
        self.frame_shape = (pm['dets_x'], pm['dets_y'])
        self.det_fname = read_config.get_filename(self.config_file, 'emc', 'in_detector_file')
        self.ewald_rad = pm['ewald_rad']
        self.detd = pm['detd']/pm['pixsize']

    def init_geom(self, mask_flag):
        if self.det_fname is None:
            self.x, self.y = np.indices(self.frame_shape)
            self.x = self.x.flatten()
            self.y = self.y.flatten()
            self.mask = np.ones(self.frame_shape)
        else:
            sys.stderr.write('Reading detector file...')
            if mask_flag:
                sys.stderr.write('with mask...')
                cx, cy, cz, mask = np.loadtxt(self.det_fname, usecols=(0,1,2,4), skiprows=1, unpack=True)
                mask[mask==2] = 1
                mask = 1 - mask
            else:
                cx, cy, cz = np.loadtxt(self.det_fname, usecols=(0,1,2), skiprows=1, unpack=True)
                mask = np.ones(cx.shape)
            sys.stderr.write('done\n')
            
            cx = cx*self.detd/(cz+self.ewald_rad)
            cy = cy*self.detd/(cz+self.ewald_rad)
            self.x = np.round(cx - cx.min()).astype('i4')
            self.y = np.round(cy - cy.min()).astype('i4')
            
            self.frame_shape = (self.x.max()+1, self.y.max()+1)
            self.mask = np.ones(self.frame_shape)
            self.mask[self.x, self.y] = mask.flatten()
            self.raw_mask = mask
            self.cx = cx
            self.cy = cy

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
        mode = self.mode_val.get()
        if num < 0 or num >= self.num_frames:
            sys.stderr.write('Frame number %d out of range!\n' % num)
            return
        
        file_num = np.where(num < self.num_data_list)[0][0]
        if file_num == 0:
            frame_num = num
        else:
            frame_num = num - self.num_data_list[file_num-1]
        frame = self.read_frame(file_num, frame_num)
        
        if mode == 2:
            s = plt.subplot(121)
            s.imshow(frame, vmin=0, vmax=float(self.rangestr.get()), interpolation='none', cmap=self.cmap)
            s.set_title("%d photons" % frame.sum())
            self.fig.add_subplot(s)
            
            s = plt.subplot(122)
            pframe = self.conversion_panel.polar.convert(frame)
            s.imshow(pframe, vmin=0, vmax=float(self.rangestr.get()), interpolation='none', cmap=self.cmap, aspect=float(pframe.shape[1])/pframe.shape[0])
            s.set_title('Polar representation')
            self.fig.add_subplot(s)
        else:
            s = plt.subplot(111)
            s.imshow(frame, vmin=0, vmax=float(self.rangestr.get()), interpolation='none', cmap=self.cmap)
            if mode == 1:
                s.set_title("%d photons (%s)" % (frame.sum(), self.manual_panel.class_list[num]))
            else:
                s.set_title("%d photons" % frame.sum())
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
        self.canvas.get_tk_widget().focus_set()

    def switch_mode(self, event=None):
        mode = self.mode_val.get()
        
        if mode != 1 and self.manual_panel is not None:
            self.manual_panel.classify_flag.set(0)
            self.manual_panel.classify_flag_changed()
            self.class_list = self.manual_panel.class_list
            self.manual_panel.grid_forget()
            self.manual_panel.destroy()
            self.manual_panel = None
        if mode != 2 and self.conversion_panel is not None:
            self.conversion_panel.grid_forget()
            self.conversion_panel.destroy()
            self.conversion_panel = None
        
        if mode == 0:
            print 'Switching to display mode'
        elif mode == 1:
            print 'Switching to manual classification mode'
            self.manual_panel = manual.Manual_panel(self, width=50)
            self.manual_panel.grid(row=0, column=1, sticky='news')
        elif mode == 2:
            print 'Switching to conversion mode'
            self.conversion_panel = conversion.Conversion_panel(self, width=30)
            self.conversion_panel.grid(row=0, column=1, sticky='news')
        self.plot_frame()

    def quit(self, event=None):
        self.master.quit()

if __name__ == '__main__':
    parser = py_utils.my_argparser(description='Utility for viewing frames of the emc file (list)')
    parser.add_argument('--cmap', help='Matplotlib color map (default: jet)')
    parser.add_argument('-M', '--mask', help='Whether to zero out masked pixels (default False)', action='store_true', default=False)
    args = parser.special_parse_args()
    
    root = Tk.Tk()
    Classifier(root, args.config_file, cmap=args.cmap, mask=args.mask)
    root.mainloop()
