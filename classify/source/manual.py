import numpy as np
import sys
import os
import string
import Tkinter as Tk
import ttk

class Manual_panel(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        ttk.Frame.__init__(self, parent.master, *args, **kwargs)
        
        self.parent = parent
        self.classify_flag = Tk.IntVar()
        self.classify_flag.set(0)
        self.class_list_fname = Tk.StringVar()
        self.class_list_fname.set('my_classes.dat')
        self.class_list_summary = Tk.StringVar()
        self.class_list = np.zeros((self.parent.num_frames,), dtype=np.str_)
        
        self.init_UI()
        self.gen_class_summary()

    def init_UI(self):
        line = ttk.Frame(self); line.pack(fill=Tk.X)
        ttk.Label(line, text='Press any [a-z] key to assign label to frame').pack(side=Tk.LEFT, fill=Tk.X)
        
        line = ttk.Frame(self); line.pack(fill=Tk.X)
        ttk.Checkbutton(line, text='Classify', variable=self.classify_flag, command=self.classify_flag_changed).pack(side=Tk.LEFT)
        ttk.Button(line, text='Unassign Class', command=self.unassign_class).pack(side=Tk.LEFT)
        
        line = ttk.Frame(self); line.pack(fill=Tk.X)
        ttk.Entry(line, textvariable=self.class_list_fname).pack(side=Tk.LEFT)
        ttk.Button(line, text='Save Class List', command=self.save_class_list).pack(side=Tk.TOP, anchor=Tk.W)
        
        line = ttk.Frame(self); line.pack(fill=Tk.X)
        #ttk.Label(line, text='Classification Summary:', font=('Helvetica', 14)).pack(side=Tk.TOP, anchor=Tk.W)
        ttk.Label(line, text='Classification Summary:').pack(side=Tk.TOP, anchor=Tk.W)
        #ttk.Label(line, textvariable=self.class_list_summary, font=('Courier', 14)).pack(side=Tk.TOP, anchor=Tk.W)
        ttk.Label(line, textvariable=self.class_list_summary).pack(side=Tk.TOP, anchor=Tk.W)

    def assign_class(self, event=None):
        num = int(self.parent.numstr.get())
        self.class_list[num] = event.char
        self.gen_class_summary()
        self.parent.next_frame()

    def unassign_class(self, event=None):
        num = int(self.parent.numstr.get())
        self.class_list[num] = ''
        self.gen_class_summary()
        self.parent.plot_frame()

    def classify_flag_changed(self, event=None):
        if self.classify_flag.get() == 0:
            for c in string.ascii_lowercase:
                self.parent.master.unbind(c)
        else:
            for c in string.ascii_lowercase:
                self.parent.master.bind(c, self.assign_class)
            self.parent.master.focus()

    def gen_class_summary(self):
        u = np.unique(self.class_list, return_counts=True)
        cmin = 0
        summary = ''
        if u[0][0] == '':
            summary += '|    |%7d|\n' % u[1][0]
            cmin = 1
        for i in range(cmin, len(u[0])):
            summary += '|%-4s|%7d|\n' % (u[0][i], u[1][i])
        self.class_list_summary.set(summary)

    def save_class_list(self, event=None):
        print 'Saving manually classified list to', self.class_list_fname.get()
        np.savetxt(self.class_list_fname.get(), self.class_list, fmt='%s')

    def read_class_list(self, event=None):
        with open(self.class_list_fname.get(), 'r') as f:
            c = np.array([l.rstrip() for l in f.readlines()])
        self.class_key, self.class_key_pos = np.unique(c, return_inverse=True)

