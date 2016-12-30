#!/usr/bin/env python

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import Tkinter as Tk
import ttk
import tkMessageBox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from source import data
from source import manual
from source import conversion
from source import embedding
from source import classes
from source import mlp
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+'/utils/py_src/')
import py_utils
import read_config

class Classifier():
    def __init__(self, master, config_file, cmap='jet', mask=False):
        self.master = master
        self.cmap = cmap
        self.config_file = config_file
        
        self.mode_val = Tk.IntVar(); self.mode_val.set(0)
        self.numstr = Tk.StringVar(); self.numstr.set(str(0))
        self.rangestr = Tk.StringVar(); self.rangestr.set(str(10))
        self.ang_corr = None
        
        self.get_config_params()
        self.init_geom(mask)
        
        self.emc_reader = data.EMC_reader(self)
        self.num_frames = self.emc_reader.num_frames
        self.classes = classes.Frame_classes(self.num_frames)
        
        self.init_UI()

    def init_UI(self):
        self.master.title('Dragonfly Diffraction Pattern Classifier')
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)
        self.master.protocol('WM_DELETE_WINDOW', self.quit)
        if sys.platform != 'darwin':
            path_string = " ".join(os.path.realpath(__file__).split('/')[:-1])
            self.master.tk.eval('source [file join / ' + path_string + ' themes plastik plastik.tcl]')
            self.master.tk.eval('source [file join / ' + path_string + ' themes clearlooks clearlooks8.5.tcl]')
            fstyle = ttk.Style()
            fstyle.theme_use('clearlooks')
            #fstyle.theme_use('plastik')
        
        fig_frame = ttk.Frame(self.master)
        fig_frame.grid(row=0, column=0, sticky='nsew')
        fig_frame.columnconfigure(0, weight=1)
        fig_frame.rowconfigure(0, weight=1)
        
        self.fig = plt.figure(figsize=(6, 6))
        #self.fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, wspace=0.0)
        self.canvas = FigureCanvasTkAgg(self.fig, fig_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas.show()
        self.canvas_widget.pack(fill='both', expand=1)
        
        self.options = ttk.Frame(self.master, relief=Tk.GROOVE, borderwidth=5, width=400, height=200)
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
        ttk.Button(line, text='Quit', command=self.quit).pack(side=Tk.RIGHT)
        
        menubar = Tk.Menu(self.master)
        modemenu = Tk.Menu(menubar, tearoff=0)
        modemenu.add_radiobutton(label='Display', underline=0, variable=self.mode_val, value=0, command=self.switch_mode)
        modemenu.add_radiobutton(label='Manual', underline=0, variable=self.mode_val, value=1, command=self.switch_mode)
        modemenu.add_radiobutton(label='Convert', underline=0, variable=self.mode_val, value=2, command=self.switch_mode)
        modemenu.add_radiobutton(label='Embedding', underline=0, variable=self.mode_val, value=3, command=self.switch_mode)
        modemenu.add_radiobutton(label='MLP', underline=1, variable=self.mode_val, value=4, command=self.switch_mode)
        menubar.add_cascade(label='Mode', menu=modemenu, underline=0)
        self.master.config(menu=menubar)
        
        self.manual_panel = manual.Manual_panel(self, width=50)
        self.conversion_panel = conversion.Conversion_panel(self, width=30)
        self.embedding_panel = embedding.Embedding_panel(self, width=30)
        self.mlp_panel = mlp.MLP_panel(self, width=30)
        
        self.master.bind('<Return>', self.plot_frame)
        self.master.bind('<KP_Enter>', self.plot_frame)
        self.master.bind('<Control-n>', self.next_frame)
        self.master.bind('<Control-p>', self.prev_frame)
        self.master.bind('<Control-r>', self.rand_frame)
        self.master.bind('<Control-q>', self.quit)
        self.canvas_widget.bind('<Button-1>', self.frame_focus)
        self.canvas_widget.bind('<Right>', self.next_frame)
        self.canvas_widget.bind('<Left>', self.prev_frame)
        self.canvas_widget.bind('<Up>', self.next_frame)
        self.canvas_widget.bind('<Down>', self.prev_frame)
        
        self.plot_frame()

    def get_config_params(self):
        try:
            pfile = read_config.get_filename(self.config_file, 'emc', 'in_photons_file')
            print 'Using in_photons_file: %s' % pfile
            self.photons_list = [pfile]
        except read_config.ConfigParser.NoOptionError:
            plist = read_config.get_filename(self.config_file, 'emc', 'in_photons_list')
            print 'Using in_photons_list: %s' % plist
            with open(plist, 'r') as f:
                self.photons_list = map(lambda x: x.rstrip(), f.readlines())
        
        pm = read_config.get_detector_config(self.config_file)
        
        self.num_files = len(self.photons_list)
        self.frame_shape = (pm['dets_x'], pm['dets_y'])
        self.det_fname = read_config.get_filename(self.config_file, 'emc', 'in_detector_file')
        self.output_folder = read_config.get_filename(self.config_file, 'emc', 'output_folder')
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

    def plot_frame(self, event=None, force_frame=False):
        mode = self.mode_val.get()
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
            s = plt.subplot(121)
            s.imshow(frame, vmin=0, vmax=float(self.rangestr.get()), interpolation='none', cmap=self.cmap)
            s.set_title("%d photons" % frame.sum())
            self.fig.add_subplot(s)
            
            s = plt.subplot(122)
            pframe = self.conversion_panel.polar.convert(frame)
            s.imshow(pframe, vmin=0, vmax=float(self.rangestr.get()), interpolation='none', cmap=self.cmap, aspect=float(pframe.shape[1])/pframe.shape[0])
            s.set_title('Polar representation')
            self.fig.add_subplot(s)
        elif (not force_frame) and mode == 3 and self.embedding_panel.embed is not None:
            plt.gcf().clear()
            for p in self.embedding_panel.roi_list:
                self.canvas_widget.tag_raise(p)
            for p in self.embedding_panel.click_points_list:
                self.canvas_widget.tag_raise(p)
            
            s = plt.subplot(111)
            e = self.embedding_panel.embed
            s.hist2d(e[:,0], e[:,1], bins=100)
            s.set_title('Spectral embedding')
            self.fig.add_subplot(s)
        else:
            if mode == 3:
                for p in self.embedding_panel.roi_list:
                    self.canvas_widget.tag_lower(p)
                for p in self.embedding_panel.click_points_list:
                    self.canvas_widget.tag_lower(p)
            plt.gcf().clear()
            s = plt.subplot(111)
            s.imshow(frame, vmin=0, vmax=float(self.rangestr.get()), interpolation='none', cmap=self.cmap)
            if mode == 1:
                s.set_title('%d photons (%s)' % (frame.sum(), self.classes.clist[num]))
            elif mode == 4 and self.mlp_panel.predictions is not None:
                s.set_title('%d photons [%s]' % (frame.sum(), self.mlp_panel.predictions[num]))
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
        self.canvas_widget.focus_set()

    def switch_mode(self, event=None):
        mode = self.mode_val.get()
        
        if mode != 1 and len(self.manual_panel.grid_info()) > 0:
            self.manual_panel.classify_flag.set(0)
            self.manual_panel.classify_flag_changed()
            self.manual_panel.grid_forget()
        if mode != 2 and len(self.conversion_panel.grid_info()) > 0:
            self.conversion_panel.grid_forget()
        if mode != 3 and len(self.embedding_panel.grid_info()) > 0:
            self.embedding_panel.grid_forget()
        if mode != 4 and len(self.mlp_panel.grid_info()) > 0:
            self.mlp_panel.grid_forget()
        
        if mode == 0:
            print 'Switching to display mode'
        elif mode == 1:
            print 'Switching to manual classification mode'
            self.manual_panel.grid(row=0, column=1, rowspan=2, sticky='news')
        elif mode == 2:
            print 'Switching to conversion mode'
            self.conversion_panel.grid(row=0, column=1, rowspan=2, sticky='news')
        elif mode == 3:
            print 'Switching to embedding mode'
            self.embedding_panel.grid(row=0, column=1, rowspan=2, sticky='news')
        elif mode == 4:
            print 'Switching to MLP mode'
            self.mlp_panel.grid(row=0, column=1, rowspan=2, sticky='news')
        self.plot_frame()

    def quit(self, event=None):
        if self.classes.unsaved:
            result = tkMessageBox.askquestion('Exit?', 'Unsaved changes to class list. Save?', parent=self.master, type=tkMessageBox.YESNOCANCEL)
            if result == 'yes':
                self.manual_panel.save_class_list()
            elif result == 'no':
                pass
            else:
                return
        
        self.master.quit()

if __name__ == '__main__':
    parser = py_utils.my_argparser(description='Utility for viewing frames of the emc file (list)')
    parser.add_argument('--cmap', help='Matplotlib color map (default: jet)')
    parser.add_argument('-M', '--mask', help='Whether to zero out masked pixels (default False)', action='store_true', default=False)
    args = parser.special_parse_args()
    
    root = Tk.Tk()
    Classifier(root, args.config_file, cmap=args.cmap, mask=args.mask)
    root.mainloop()
