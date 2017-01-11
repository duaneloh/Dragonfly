#!/usr/bin/env python

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import Tkinter as Tk
import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from py_src import py_utils
from py_src import read_config
from py_src import frame_panel
from py_src import data

class Frameviewer():
    def __init__(self, master, config_file, cmap='jet', mask=False):
        self.master = master
        self.cmap = cmap
        self.config_file = config_file
        self.mode_val = None
        
        self.get_config_params()
        self.geom = data.Det_reader(self.det_fname, self.detd, self.ewald_rad, mask_flag=mask)
        self.emc_reader = data.EMC_reader(self.photons_list, self.geom.x, self.geom.y, self.geom.mask)
        self.num_frames = self.emc_reader.num_frames
        self.init_UI()

    def init_UI(self):
        self.master.title('Dragonfly Frame Viewer')
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
        output_folder = read_config.get_filename(self.config_file, 'emc', 'output_folder')
        try:
            self.blacklist = np.loadtxt(read_config.get_filename(args.config_file, 'emc', 'blacklist_file'), dtype='u1')
        except read_config.ConfigParser.NoOptionError:
            self.blacklist = None
        
        self.output_folder = os.path.realpath(output_folder)
        self.ewald_rad = pm['ewald_rad']
        self.detd = pm['detd']/pm['pixsize']

    def next_bad_frame(self, event=None):
        cur_num = int(self.numstr.get())
        ind = np.where(self.blacklist==1)[0]
        num = ind[ind>cur_num][0]
        #num = np.where(np.where(self.blacklist==1)[0]>cur_num)[0][0]
        self.numstr.set(str(num))
        self.plot_frame()

    def quit(self, event=None):
        self.master.quit()

if __name__ == '__main__':
    parser = py_utils.my_argparser(description='Utility for viewing frames of the emc file (list)')
    parser.add_argument('--cmap', help='Matplotlib color map (default: jet)')
    parser.add_argument('-M', '--mask', help='Whether to zero out masked pixels (default False)', action='store_true', default=False)
    args = parser.special_parse_args()
    
    root = Tk.Tk()
    Frameviewer(root, args.config_file, cmap=args.cmap, mask=args.mask)
    root.mainloop()
