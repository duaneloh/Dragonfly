import numpy as np
import sys
import os
import string
import Tkinter as Tk

class Embedding_panel(Tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        Tk.Frame.__init__(self, parent.master, *args, **kwargs)
        self.parent = parent
        self.init_UI()

    def init_UI(self):
        line = Tk.Frame(self); line.pack(fill=Tk.X)
        Tk.Label(line, text='Spectral manifold embedding').pack(side=Tk.LEFT, fill=Tk.X)
        
        line = Tk.Frame(self); line.pack(fill=Tk.X)
        Tk.Label(line, text='Frame range:').pack(side=Tk.LEFT)
        Tk.Entry(line, textvariable=self.parent.conversion_panel.first_frame, width=8).pack(side=Tk.LEFT)
        Tk.Label(line, text='-').pack(side=Tk.LEFT)
        Tk.Entry(line, textvariable=self.parent.conversion_panel.last_frame, width=8).pack(side=Tk.LEFT)
