#!/usr/bin/env python

import sys
import os
import numpy as np
import Tkinter as Tk
import ttk
import tkMessageBox
from py_src import data
from py_src import manual
from py_src import conversion
from py_src import embedding
from py_src import classes
from py_src import mlp
from py_src import py_utils
from py_src import read_config
from py_src import frame_panel

class Classifier():
    def __init__(self, master, config_file, cmap='jet', mask=False):
        self.master = master
        self.cmap = cmap
        self.config_file = config_file
        self.mode_val = Tk.IntVar(); self.mode_val.set(0)
        self.ang_corr = None
        
        self.get_config_params()
        self.geom = data.Det_reader(self.det_fname, self.detd, self.ewald_rad, mask_flag=mask)
        self.emc_reader = data.EMC_reader(self.photons_list, self.geom.x, self.geom.y, self.geom.mask)
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
        
        self.frame_panel = frame_panel.Frame_panel(self)
        self.frame_panel.grid(row=0, column=0, sticky='nsew')
        self.frame_panel.columnconfigure(0, weight=1)
        self.frame_panel.rowconfigure(0, weight=1)
        self.frame_panel.mode = self.mode_val
        
        self.manual_panel = manual.Manual_panel(self, width=50)
        self.conversion_panel = conversion.Conversion_panel(self, width=30)
        self.embedding_panel = embedding.Embedding_panel(self, width=30)
        self.mlp_panel = mlp.MLP_panel(self, width=30)
        
        menubar = Tk.Menu(self.master)
        modemenu = Tk.Menu(menubar, tearoff=0)
        modemenu.add_radiobutton(label='Display', underline=0, variable=self.mode_val, value=0, command=self.switch_mode)
        modemenu.add_radiobutton(label='Manual', underline=0, variable=self.mode_val, value=1, command=self.switch_mode)
        modemenu.add_radiobutton(label='Convert', underline=0, variable=self.mode_val, value=2, command=self.switch_mode)
        modemenu.add_radiobutton(label='Embedding', underline=0, variable=self.mode_val, value=3, command=self.switch_mode)
        modemenu.add_radiobutton(label='MLP', underline=1, variable=self.mode_val, value=4, command=self.switch_mode)
        menubar.add_cascade(label='Mode', menu=modemenu, underline=0)
        self.master.config(menu=menubar)

    def get_config_params(self):
        try:
            pfile = read_config.get_filename(self.config_file, 'classifier', 'in_photons_file')
            print 'Using in_photons_file: %s' % pfile
            self.photons_list = [pfile]
        except read_config.ConfigParser.NoOptionError:
            plist = read_config.get_filename(self.config_file, 'classifier', 'in_photons_list')
            print 'Using in_photons_list: %s' % plist
            with open(plist, 'r') as f:
                self.photons_list = map(lambda x: x.rstrip(), f.readlines())
        
        pm = read_config.get_detector_config(self.config_file)
        
        self.num_files = len(self.photons_list)
        self.frame_shape = (pm['dets_x'], pm['dets_y'])
        self.det_fname = read_config.get_filename(self.config_file, 'classifier', 'in_detector_file')
        output_folder = read_config.get_filename(self.config_file, 'classifier', 'output_folder')
        self.output_folder = os.path.realpath(output_folder)
        self.ewald_rad = pm['ewald_rad']
        self.detd = pm['detd']/pm['pixsize']

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
            pass
        elif mode == 1:
            self.manual_panel.grid(row=0, column=1, rowspan=2, sticky='news')
        elif mode == 2:
            self.conversion_panel.grid(row=0, column=1, rowspan=2, sticky='news')
        elif mode == 3:
            self.embedding_panel.grid(row=0, column=1, rowspan=2, sticky='news')
        elif mode == 4:
            self.mlp_panel.grid(row=0, column=1, rowspan=2, sticky='news')
        self.frame_panel.plot_frame()

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
