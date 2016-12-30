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
        self.emc_reader = self.parent.emc_reader
        
        self.layer_sizes = Tk.StringVar(); self.layer_sizes.set('10, 10')
        self.alpha_var = Tk.StringVar(); self.alpha_var.set('1.e-3')
        self.predict_summary = Tk.StringVar(); self.predict_summary.set('')
        self.predictions = None
        
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
        
        line = ttk.Frame(self); line.pack(fill=Tk.X)
        ttk.Button(line, text='Train', command=self.train).pack(side=Tk.LEFT)

    def remake_mlp(self, event=None):
        sizes = tuple([int(s.strip()) for s in self.layer_sizes.get().split(',')])
        alpha = float(self.alpha_var.get())
        self.mlp = neural_network.MLPClassifier(hidden_layer_sizes=sizes, alpha=alpha)

    def train(self, event=None):
        self.get_training_data()
        self.mlp.fit(self.train_data, self.train_labels)
        print 'Done training'
        #print 'Score on training set =', self.mlp.score(self.train_data, self.train_labels)
        self.add_predict_frame()

    def get_training_data(self):
        ang_corr = self.parent.ang_corr
        if ang_corr is None:
            #self.parent.conversion_panel.convert_frames()
            self.parent.ang_corr = np.load('data/ang_corr.npy') #FIXME For debugging
            ang_corr = self.parent.ang_corr
        ang_corr = ang_corr.reshape(-1, ang_corr.shape[1]*ang_corr.shape[2])
        
        key_pos = self.classes.key_pos[int(self.parent.conversion_panel.first_frame.get()):int(self.parent.conversion_panel.last_frame.get())]
        self.train_data = ang_corr[key_pos>0]
        self.train_labels = key_pos[key_pos>0]

    def add_predict_frame(self):
        self.predict_frame = ttk.Frame(self); self.predict_frame.pack(fill=Tk.X)
        
        line = ttk.Frame(self.predict_frame); line.pack(fill=Tk.X)
        ttk.Button(line, text='Predict', command=self.predict).pack(side=Tk.LEFT)
        
        line = ttk.Frame(self.predict_frame); line.pack(fill=Tk.X)
        ttk.Label(line, textvariable=self.predict_summary).pack(fill=Tk.X)

    def predict(self, event=None):
        self.predictions = np.zeros((self.parent.num_frames,), dtype=np.str_)
        for i in range(len(self.predictions)):
            self.parent.conversion_panel.polar.convert(self.emc_reader.get_frame(i))
            ang_corr = np.expand_dims(self.parent.conversion_panel.polar.compute_ang_corr().flatten(), axis=0)
            self.predictions[i] = self.classes.key[self.mlp.predict(ang_corr)[0]]
            sys.stderr.write('\r%d/%d' % (i+1, self.parent.num_frames))
        self.gen_predict_summary()

    def gen_predict_summary(self, event=None):
        summary=''
        key, counts = np.unique(self.predictions, return_counts=True)
        for i, c in zip(key, counts):
            summary += '|%-4s|%7d|\n' % (i, c)
        self.predict_summary.set(summary)
