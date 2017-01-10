import numpy as np
import sys
import os
import string
import Tkinter as Tk
import ttk
from sklearn import neural_network
import multiprocessing
import ctypes

class MLP_panel(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        ttk.Frame.__init__(self, parent.master, *args, **kwargs)
        self.parent = parent
        self.classes = self.parent.classes
        self.emc_reader = self.parent.emc_reader
        self.conversion = self.parent.conversion_panel
        
        self.layer_sizes = Tk.StringVar(); self.layer_sizes.set('10, 10')
        self.alpha_var = Tk.StringVar(); self.alpha_var.set('1.e-5')
        self.predict_summary = Tk.StringVar(); self.predict_summary.set('')
        self.predictions_fname = Tk.StringVar(); self.predictions_fname.set('predictions.dat')
        self.predict_first = Tk.StringVar(); self.predict_first.set('0')
        self.predict_last = Tk.StringVar(); self.predict_last.set('1000')
        self.num_proc = Tk.StringVar(); self.num_proc.set('1')
        self.predictions = None
        self.trained = False
        
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
        entry.bind('<KP_Enter>', self.remake_mlp)
        
        line = ttk.Frame(self); line.pack(fill=Tk.X)
        ttk.Label(line, text='alpha').pack(side=Tk.LEFT)
        entry = ttk.Entry(line, textvariable=self.alpha_var, width=5)
        entry.pack(side=Tk.LEFT)
        entry.bind('<Return>', self.remake_mlp)
        entry.bind('<KP_Enter>', self.remake_mlp)
        
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
        if not self.trained:
            self.add_predict_frame()
        self.trained = True

    def get_training_data(self):
        ang_corr = self.parent.ang_corr
        if ang_corr is None:
            #self.conversion.convert_frames()
            self.parent.ang_corr = np.load(self.parent.output_folder+'/ang_corr.npy') #FIXME For debugging
            ang_corr = self.parent.ang_corr
        
        key_pos = self.classes.key_pos[int(self.conversion.first_frame.get()):int(self.conversion.last_frame.get())]
        self.train_data = ang_corr[key_pos>0]
        self.train_labels = key_pos[key_pos>0]

    def add_predict_frame(self):
        self.predict_frame = ttk.Frame(self); self.predict_frame.pack(fill=Tk.X)
        
        line = ttk.Frame(self.predict_frame); line.pack(fill=Tk.X)
        ttk.Entry(line, textvariable=self.predict_first, width=5).pack(side=Tk.LEFT)
        ttk.Entry(line, textvariable=self.predict_last, width=5).pack(side=Tk.LEFT)
        
        line = ttk.Frame(self.predict_frame); line.pack(fill=Tk.X)
        ttk.Button(line, text='Predict', command=self.predict).pack(side=Tk.LEFT)
        ttk.Entry(line, textvariable=self.num_proc, width=2).pack(side=Tk.LEFT)
        
        line = ttk.Frame(self.predict_frame); line.pack(fill=Tk.X)
        ttk.Label(line, textvariable=self.predict_summary).pack(side=Tk.LEFT)
        
        line = ttk.Frame(self.predict_frame); line.pack(fill=Tk.X)
        ttk.Entry(line, textvariable=self.predictions_fname).pack(side=Tk.LEFT)
        ttk.Button(line, text='Save', command=self.save_predictions).pack(side=Tk.TOP, anchor=Tk.W)

    def predict(self, event=None):
        try:
            first = int(self.predict_first.get())
            last = int(self.predict_last.get())
            num_proc = int(self.num_proc.get())
        except ValueError:
            print 'Integers only'
            return
        
        if last < 0:
            last = self.parent.num_frames
        
        predictions = multiprocessing.Array(ctypes.c_char, self.parent.num_frames)
        jobs = []
        for i in range(num_proc):
            p = multiprocessing.Process(target=self.predict_worker, args=(i, num_proc, np.arange(first, last, dtype='i4'), predictions))
            jobs.append(p)
            p.start()
        for j in jobs:
            j.join()
        sys.stderr.write('\r%d/%d\n' % (last, last))
        
        self.predictions = np.frombuffer(predictions.get_obj(), dtype='S1')
        self.gen_predict_summary()

    def get_and_convert(self, num):
        return self.conversion.polar.compute_ang_corr(self.conversion.polar.convert(self.emc_reader.get_frame(num)))

    def predict_worker(self, rank, num_proc, indices, predictions):
        my_ind = indices[rank::num_proc]
        for i in my_ind:
            ang_corr = np.expand_dims(self.get_and_convert(i).flatten(), axis=0)
            predictions[i] = self.classes.key[self.mlp.predict(ang_corr)[0]]
            if rank == 0:
                sys.stderr.write('\r%d/%d'%(i, indices[-1]))

    def gen_predict_summary(self, event=None):
        summary=''
        key, counts = np.unique(self.predictions, return_counts=True)
        for i, c in zip(key, counts):
            summary += '|%-4s|%7d|\n' % (i, c)
        self.predict_summary.set(summary)

    def save_predictions(self, event=None):
        print 'Saving predictions list to', self.predictions_fname.get()
        np.savetxt(self.predictions_fname.get(), self.predictions, fmt='%s')
