import numpy as np
import sys
import os
import string
import Tkinter as Tk
import ttk
from sklearn import neural_network

class MLP_panel(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        ttk.Frame.__init__(self, parent.master, *args, **kwargs)
        self.parent = parent
        self.classes = self.parent.classes
        
        self.layer_sizes = Tk.StringVar(); self.layer_sizes.set('10, 10')
        self.alpha_var = Tk.StringVar(); self.alpha_var.set('1.e-3')
        
        self.init_UI()
        self.remake_mlp()
        
    def init_UI(self):
        line = ttk.Frame(self); line.pack(fill=Tk.X)
        ttk.Label(line, text='Multi-Layer Perceptron').pack(side=Tk.LEFT, fill=Tk.X)
        
        line = ttk.Frame(self); line.pack(fill=Tk.X)
        ttk.Label(line, text='Hidden layer sizes').pack(side=Tk.LEFT)
        entry = ttk.Entry(line, textvariable=self.layer_sizes, width=10)
        entry.pack(side=Tk.LEFT)
        entry.bind('<Return>', self.remake_mlp)
        
        line = ttk.Frame(self); line.pack(fill=Tk.X)
        ttk.Label(line, text='alpha').pack(side=Tk.LEFT)
        entry = ttk.Entry(line, textvariable=self.alpha_var, width=5)
        entry.pack(side=Tk.LEFT)
        entry.bind('<Return>', self.remake_mlp)
        
        line = ttk.Frame(self); line.pack(fill=Tk.X)
        ttk.Button(line, text='Update', command=self.remake_mlp).pack(side=Tk.LEFT)

    def remake_mlp(self, event=None):
        sizes = tuple([int(s.strip()) for s in self.layer_sizes.get().split(',')])
        alpha = float(self.alpha_var.get())
        self.mlp = neural_network.MLPClassifier(hidden_layer_sizes=sizes, alpha=alpha)
