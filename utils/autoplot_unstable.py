#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys
import Tkinter as Tk
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import time
from glob import glob
import re

class Plotter:
    def __init__(self, master, size=99):
        self.master = master
        self.size = size
        self.center = self.size/2
        self.max_iter = 0

        self.fname = Tk.StringVar()
        self.logfname = Tk.StringVar()
        self.rangestr = Tk.StringVar()
        self.imagename = Tk.StringVar()
        self.layernum = Tk.IntVar()
        self.ifcheck = Tk.IntVar()
        self.iter = Tk.IntVar()
        self.orientnum = set()
        self.orient = []

        self.fname.set('data/output/intens_001.bin')
        self.logfname.set('EMC.log')
        self.imagename.set('images/' + os.path.splitext(os.path.basename(self.fname.get()))[0] + '.png')
        self.image_exists = False
        self.rangestr.set(str(1.))
        self.layernum.set(self.center)
        self.ifcheck.set(0)
        self.iter.set(0)

        self.fig = plt.figure(figsize=(14,5))
        self.fig.subplots_adjust(left=0.0, bottom=0.00, right=0.99, wspace=0.0)
        self.canvas = FigureCanvasTkAgg(self.fig, self.master)
        self.canvas.get_tk_widget().grid()

        self.log_fig = plt.figure(figsize=(14,5), facecolor='white')
        #self.log_fig.subplots_adjust(left=0.0, bottom=0.00, right=0.99, wspace=0.0)
        self.plotcanvas = FigureCanvasTkAgg(self.log_fig, self.master)
        self.plotcanvas.get_tk_widget().grid(row=1)

        self.options = Tk.Frame(self.master,relief=Tk.GROOVE,borderwidth=5,width=300, height=200)
        #self.options.grid(row=0,column=1,rowspan=2,sticky=Tk.N+Tk.S)
        self.options.grid(row=0,column=1,rowspan=2,sticky=Tk.N)

        self.old_fname = self.fname.get()
        self.old_rangestr = self.rangestr.get()

        self.master.bind('<Return>', self.parse_and_plot)
        self.master.bind('<KP_Enter>', self.parse_and_plot)
        self.master.bind('<Control-s>', self.save_plot)
        self.master.bind('<Control-q>', self.quit_)
        self.master.bind('<Up>', self.increment_layer)
        self.master.bind('<Down>', self.decrement_layer)

        self.init_UI()

    def init_UI(self):
        line = Tk.Frame(self.options)
        line.pack(fill=Tk.X)
        Tk.Label(line,text="Log Filename: ").pack(side=Tk.LEFT)
        Tk.Entry(line,textvariable=self.logfname,width=20).pack(side=Tk.LEFT)
        Tk.Label(line,text="PlotMax: ").pack(side=Tk.LEFT)
        Tk.Entry(line,textvariable=self.rangestr,width=10).pack(side=Tk.LEFT)

        line = Tk.Frame(self.options)
        line.pack(fill=Tk.X)
        Tk.Label(line,text="Filename: ").pack(side=Tk.LEFT)
        Tk.Entry(line,textvariable=self.fname,width=40).pack(side=Tk.LEFT)

        line = Tk.Frame(self.options)
        line.pack(fill=Tk.X)
        Tk.Label(line,text="Image name: ").pack(side=Tk.LEFT)
        Tk.Entry(line,textvariable=self.imagename,width=30).pack(side=Tk.LEFT)
        Tk.Button(line,text="Save",command=self.save_plot).pack(side=Tk.LEFT)

        line = Tk.Frame(self.options)
        line.pack()
        Tk.Label(line,text='Layer no. ').pack(side=Tk.LEFT)
        Tk.Button(line,text="-",command=self.decrement_layer).pack(side=Tk.LEFT,fill=Tk.Y)
        Tk.Scale(line,from_=0,to=int(self.size),orient=Tk.HORIZONTAL,length=250,width=20,
                 variable=self.layernum).pack(side=Tk.LEFT)
        Tk.Button(line,text="+",command=self.increment_layer).pack(side=Tk.LEFT,fill=Tk.Y)

        line = Tk.Frame(self.options)
        line.pack()
        Tk.Label(line,text='Iteration: ').pack(side=Tk.LEFT)
        Tk.Button(line,text="-",command=self.decrement_iter).pack(side=Tk.LEFT,fill=Tk.Y)
        self.slider = Tk.Scale(line,from_=0,to=self.max_iter,orient=Tk.HORIZONTAL,length=250,width=20,
                               variable=self.iter,command=self.change_iter)
        self.slider.pack(side=Tk.LEFT)
        Tk.Button(line,text="+",command=self.increment_iter).pack(side=Tk.LEFT,fill=Tk.Y)

        line = Tk.Frame(self.options)
        line.pack(fill=Tk.X)
        Tk.Button(line,text="Check",command=self.check_for_new).pack(side=Tk.LEFT)
        Tk.Checkbutton(line,text="Keep checking",variable=self.ifcheck,command=self.keep_checking).pack(side=Tk.LEFT)

        line = Tk.Frame(self.options)
        line.pack(fill=Tk.X)
        Tk.Button(line,text="Quit",command=self.master.quit).pack(side=Tk.RIGHT)
        Tk.Button(line,text="Reparse",command=self.force_plot).pack(side=Tk.RIGHT)
        Tk.Button(line,text="Plot",command=self.parse_and_plot).pack(side=Tk.RIGHT)

    def plot_vol(self, num):
        self.imagename.set('images/' + os.path.splitext(os.path.basename(self.fname.get()))[0] + '.png')
        rangemax = float(self.rangestr.get())

        a = self.vol[num,:,:]**0.2
        b = self.vol[:,num,:]**0.2
        c = self.vol[:,:,num]**0.2

        self.fig.clf()
        grid = gridspec.GridSpec(1,3, wspace=0., hspace=0.)

        s1 = plt.Subplot(self.fig, grid[:,0])
        s1.imshow(a, vmin=0, vmax=rangemax, cmap='jet', interpolation='none')
        s1.set_title("YZ plane", y = 1.01)
        s1.axis('off')
        self.fig.add_subplot(s1)

        s2 = plt.Subplot(self.fig, grid[:,1])
        s2.matshow(b, vmin=0, vmax=rangemax, cmap='jet', interpolation='none')
        s2.set_title("XZ plane", y = 1.01)
        s2.axis('off')
        self.fig.add_subplot(s2)

        s3 = plt.Subplot(self.fig, grid[:,2])
        s3.matshow(c, vmin=0, vmax=rangemax, cmap='jet', interpolation='none')
        s3.set_title("XY plane", y = 1.01)
        s3.axis('off')
        self.fig.add_subplot(s3)

        self.canvas.show()

        self.image_exists = True
        self.old_rangestr = self.rangestr.get()

    def parse(self):
        s = int(self.size)
        fname = self.fname.get()

        if os.path.isfile(fname):
            f = open(fname, "r")
        else:
            print "Unable to open", fname
            return

        #self.vol = np.fromfile(f, dtype='f8', count=s*s*s).reshape((s,s,s))
        self.vol = np.fromfile(f, dtype='f8')
        self.size = int(np.ceil(np.power(len(self.vol), 1./3.)))
        self.vol = self.vol.reshape(self.size, self.size, self.size)
        self.center = self.size/2
        if not self.image_exists:
            self.layernum.set(self.center)

        self.old_fname = fname

    def plot_log(self):
        with open(self.logfname.get(), 'r') as f:
            lines = [l.rstrip().split() for l in f.readlines()]
            flag = False
            loglines = []
            for l in lines:
                if len(l) < 1:
                    continue
                if flag == True:
                    loglines.append(l)
                elif l[0] == 'Iteration':
                    flag = True
        o_files = sorted(glob("data/orientations/*.dat"))
        for p in o_files:
            fn = os.path.split(p)[-1]
            label = int(re.search("orientations_(\d+).dat", fn).groups(1)[0])
            if label not in self.orientnum:
                print "reading",  fn
                self.orientnum.add(label)
                with open(p, 'r') as f:
                    self.orient.append(np.asarray([int(l.rstrip()) for l in f.readlines()]))
            else:
                print "skipping", label

        o_array = np.asarray(self.orient)
        ord = o_array[-1].argsort()
        for index in range(len(o_array)):
            o_array[index] = o_array[index][ord]
        o_array = o_array.T

        loglines = np.array(loglines)
        if len(loglines) == 0:
            return
        iter = loglines[:,0].astype(np.int32)
        change = loglines[:,3].astype(np.float64)
        info = loglines[:,4].astype(np.float64)
        like = loglines[:,5].astype(np.float64)

        self.log_fig.clf()
        grid = gridspec.GridSpec(2,3, wspace=0.3, hspace=0.2)
        grid.update(left=0.05, right=0.99, hspace=0.0, wspace=0.2)

        s1 = plt.Subplot(self.log_fig, grid[:,0])
        s1.plot(iter, change, 'o-')
        s1.set_yscale('log')
        s1.set_xlabel('Iteration')
        s1.set_ylabel('RMS change')
        self.log_fig.add_subplot(s1)

        s2 = plt.Subplot(self.log_fig, grid[0,1])
        s2.plot(iter, info, 'o-')
        s2.get_xaxis().set_ticks([])
        s2.set_ylabel(r'Mutual info. $I(K,\theta)$')
        self.log_fig.add_subplot(s2)

        s3 = plt.Subplot(self.log_fig, grid[1,1])
        s3.plot(iter, like, 'o-')
        s3.set_xlabel('Iteration')
        s3.set_ylabel('Avg log-likelihood')
        self.log_fig.add_subplot(s3)

        s4 = plt.Subplot(self.log_fig, grid[:,2])
        sh = o_array.shape
        s4.imshow(o_array, aspect=(1.*sh[1]/sh[0]))
        s4.get_yaxis().set_ticks([])
        s4.set_xlabel('Iteration')
        s4.set_ylabel('Most likely orientations')
        self.log_fig.add_subplot(s4)

        #s1 = self.log_fig.add_subplot(131)
        #s1.plot(iter, change, 'o-')
        #s1.set_yscale('log')
        #s1.set_xlabel('Iteration')
        #s1.set_ylabel('RMS change')
        #s2 = self.log_fig.add_subplot(132)
        #s2.plot(iter, info, 'o-')
        #s2.set_xlabel('Iteration')
        #s2.set_ylabel(r'Mutual info. $I(K,\theta)$')
        #s3 = self.log_fig.add_subplot(133)
        #s3.plot(iter, like, 'o-')
        #s3.set_xlabel('Iteration')
        #s3.set_ylabel('Avg log-likelihood')
        #self.log_fig.tight_layout()

        self.plotcanvas.show()

    def parse_and_plot(self, event=None):
        if not self.image_exists:
            self.parse()
            self.plot_vol(self.layernum.get())
        elif self.old_fname == self.fname.get() and self.old_rangestr != self.rangestr.get():
            self.plot_vol(self.layernum.get())
        else:
            self.parse()
            self.plot_vol(self.layernum.get())

    def check_for_new(self, event=None):
        with open(self.logfname.get(), 'r') as f:
            last_line = f.readlines()[-1].rstrip().split()
        try:
            iter = int(last_line[0])
        except ValueError:
            iter = 0

        if iter > 0 and self.max_iter != iter:
            self.fname.set('data/output/intens_%.3d.bin' % iter)
            self.max_iter = iter
            self.slider.configure(to=self.max_iter)
            self.iter.set(iter)
            self.plot_log()
            self.parse_and_plot()

    def keep_checking(self, event=None):
        if self.ifcheck.get() is 1:
            self.check_for_new()
            self.master.after(5000, self.keep_checking)

    def force_plot(self, event=None):
        self.parse()
        self.plot_vol(self.layernum.get())

    def increment_layer(self, event=None):
        self.layernum.set(min(self.layernum.get()+1, self.size-1))
        self.plot_vol(self.layernum.get())

    def decrement_layer(self, event=None):
        self.layernum.set(max(self.layernum.get()-1, 0))
        self.plot_vol(self.layernum.get())

    def increment_iter(self, event=None):
        self.iter.set(min(self.iter.get()+1, self.max_iter))
        if self.iter.get() >= 0:
            self.fname.set('data/output/intens_%.3d.bin' % self.iter.get())
            self.parse_and_plot()

    def decrement_iter(self, event=None):
        self.iter.set(max(self.iter.get()-1, 0))
        if self.iter.get() >= 0:
            self.fname.set('data/output/intens_%.3d.bin' % self.iter.get())
            self.parse_and_plot()

    def change_iter(self, event=None):
        if self.iter.get() >= 0:
            self.fname.set('data/output/intens_%.3d.bin' % self.iter.get())

    def save_plot(self, event=None):
        self.fig.savefig(self.imagename.get(), bbox_inches='tight')
        print "Saved to", self.imagename.get()

    def quit_(self, event=None):
        self.master.quit()

root = Tk.Tk()
plotter = Plotter(root, 210)
root.mainloop()
