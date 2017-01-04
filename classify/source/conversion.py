import numpy as np
import sys
import os
import string
import Tkinter as Tk
import ttk
import polar
import multiprocessing
import ctypes

class Conversion_panel(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        ttk.Frame.__init__(self, parent.master, *args, **kwargs)
        
        self.parent = parent
        self.emc_reader = self.parent.emc_reader
        self.r_min = Tk.StringVar(); self.r_min.set('16')
        self.r_max = Tk.StringVar(); self.r_max.set('80')
        self.delta_r = Tk.StringVar(); self.delta_r.set('2')
        self.delta_ang = Tk.StringVar(); self.delta_ang.set('10')
        self.first_frame = Tk.StringVar(); self.first_frame.set('0')
        self.last_frame = Tk.StringVar(); self.last_frame.set('1000')
        self.save_flag = Tk.IntVar(); self.save_flag.set(1)
        self.num_proc = Tk.StringVar(); self.num_proc.set('1')
        
        self.polar = None
        self.init_UI()
        self.remake_converter(replot=False)

    def init_UI(self):
        line = ttk.Frame(self); line.pack(fill=Tk.X)
        ttk.Label(line, text='Convert to angular correlations').pack(side=Tk.LEFT)
        
        line = ttk.Frame(self); line.pack(fill=Tk.X)
        ttk.Label(line, text='R_min:').pack(side=Tk.LEFT)
        entry = ttk.Entry(line, textvariable=self.r_min, width=5)
        entry.pack(side=Tk.LEFT)
        entry.bind('<Return>', self.remake_converter)
        ttk.Label(line, text='R_max:').pack(side=Tk.LEFT)
        entry = ttk.Entry(line, textvariable=self.r_max, width=5)
        entry.pack(side=Tk.LEFT)
        entry.bind('<Return>', self.remake_converter)
        
        line = ttk.Frame(self); line.pack(fill=Tk.X)
        ttk.Label(line, text='dR:').pack(side=Tk.LEFT)
        entry = ttk.Entry(line, textvariable=self.delta_r, width=5)
        entry.pack(side=Tk.LEFT)
        entry.bind('<Return>', self.remake_converter)
        ttk.Label(line, text=u'd\u03b8:').pack(side=Tk.LEFT)
        entry = ttk.Entry(line, textvariable=self.delta_ang, width=5)
        entry.pack(side=Tk.LEFT)
        entry.bind('<Return>', self.remake_converter)
        ttk.Label(line, text='deg').pack(side=Tk.LEFT)
        
        line = ttk.Frame(self); line.pack(fill=Tk.X)
        ttk.Button(line, text='Update', command=self.remake_converter).pack(side=Tk.LEFT)
        
        line = ttk.Frame(self); line.pack(fill=Tk.X)
        ttk.Label(line, text='Batch processing').pack(side=Tk.LEFT)
        
        line = ttk.Frame(self); line.pack(fill=Tk.X)
        ttk.Label(line, text='Frame range:').pack(side=Tk.LEFT)
        ttk.Entry(line, textvariable=self.first_frame, width=8).pack(side=Tk.LEFT)
        ttk.Label(line, text='-').pack(side=Tk.LEFT)
        ttk.Entry(line, textvariable=self.last_frame, width=8).pack(side=Tk.LEFT)
        
        line = ttk.Frame(self); line.pack(fill=Tk.X)
        ttk.Button(line, text='Process', command=self.convert_frames).pack(side=Tk.LEFT)
        ttk.Entry(line, textvariable=self.num_proc, width=2).pack(side=Tk.LEFT)
        ttk.Checkbutton(line, text='Save to file', variable=self.save_flag).pack(side=Tk.RIGHT)

    def remake_converter(self, replot=True, event=None):
        self.polar = polar.Polar_converter(self.parent.cx, 
                                               self.parent.cy, 
                                               self.parent.raw_mask,
                                               r_min = float(self.r_min.get()),
                                               r_max = float(self.r_max.get()),
                                               delta_r = float(self.delta_r.get()),
                                               delta_ang = float(self.delta_ang.get()))
        if replot:
            self.parent.plot_frame()

    def convert_frames(self, save=True, event=None):
        ang_corr = []
        try:
            start = int(self.first_frame.get())
            end = int(self.last_frame.get())
            num_proc = int(self.num_proc.get())
        except ValueError:
            print 'Integers only'
            return
        
        arr = self.get_and_convert(0)
        ang_corr = multiprocessing.Array(ctypes.c_double, arr.size*(end-start))
        jobs = []
        for i in range(num_proc):
            p = multiprocessing.Process(target=self.convert_worker, args=(i, num_proc, np.arange(start, end, dtype='i4'), arr.size, ang_corr))
            jobs.append(p)
            p.start()
        for j in jobs:
            j.join()
        sys.stderr.write('\r%d/%d\n' % (end, end))
        
        self.parent.ang_corr = np.frombuffer(ang_corr.get_obj()).reshape(end-start, -1)
        if save:
            fname = self.parent.output_folder + '/ang_corr.npy'
            print 'Saving angular correlations to', fname
            np.save(fname, self.parent.ang_corr)

    def get_and_convert(self, num):
        return self.polar.compute_ang_corr(self.polar.convert(self.emc_reader.get_frame(num)))

    def convert_worker(self, rank, num_proc, indices, size, ang_corr):
        my_ind = indices[rank::num_proc]
        np_ang_corr = np.frombuffer(ang_corr.get_obj())
        for i in my_ind:
            ang_ind = np.where(indices==i)[0][0]
            np_ang_corr[size*ang_ind:size*(ang_ind+1)] = self.get_and_convert(i).flatten()
            if rank == 0:
                sys.stderr.write('\r%d/%d'%(i, indices[-1]))

