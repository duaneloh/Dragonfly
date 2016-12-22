import numpy as np
import sys
import os
import string
import Tkinter as Tk
import polar

class Conversion_panel(Tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        Tk.Frame.__init__(self, parent.master, *args, **kwargs)
        
        self.parent = parent
        self.r_min = Tk.StringVar()
        self.r_min.set('5')
        self.r_max = Tk.StringVar()
        self.r_max.set('60')
        self.delta_r = Tk.StringVar()
        self.delta_r.set('2')
        self.delta_ang = Tk.StringVar()
        self.delta_ang.set('10')
        
        self.polar = None
        self.init_UI()
        self.remake_converter(replot=False)

    def init_UI(self):
        line = Tk.Frame(self); line.pack(fill=Tk.X)
        Tk.Label(line, text='Convert to angular correlations').pack(side=Tk.LEFT)
        
        line = Tk.Frame(self); line.pack(fill=Tk.X)
        Tk.Label(line, text='R_min:').pack(side=Tk.LEFT)
        Tk.Entry(line, textvariable=self.r_min).pack(side=Tk.LEFT)
        Tk.Label(line, text='R_max:').pack(side=Tk.LEFT)
        Tk.Entry(line, textvariable=self.r_max).pack(side=Tk.LEFT)
        
        line = Tk.Frame(self); line.pack(fill=Tk.X)
        Tk.Label(line, text='dR:').pack(side=Tk.LEFT)
        Tk.Entry(line, textvariable=self.delta_r).pack(side=Tk.LEFT)
        Tk.Label(line, text='dtheta:').pack(side=Tk.LEFT)
        Tk.Entry(line, textvariable=self.delta_ang).pack(side=Tk.LEFT)
        Tk.Label(line, text='deg').pack(side=Tk.LEFT)
        
        line = Tk.Frame(self); line.pack(fill=Tk.X)
        Tk.Button(line, text='Update', command=self.remake_converter).pack(side=Tk.LEFT)

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
