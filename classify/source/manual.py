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
        self.classes = self.parent.classes
        self.numstr = self.parent.numstr
        self.classify_flag = Tk.IntVar(); self.classify_flag.set(0)
        self.class_list_fname = Tk.StringVar(); self.class_list_fname.set('my_classes.dat')
        self.class_num = Tk.IntVar(); self.class_num.set(-1)
        
        self.classes.init_list(fname=self.class_list_fname.get())
        self.class_list_summary = Tk.StringVar(); self.class_list_summary.set(self.classes.gen_summary())
        
        self.init_UI()

    def init_UI(self):
        line = ttk.Frame(self); line.pack(fill=Tk.X)
        ttk.Label(line, text='Press any [a-z] key to assign label to frame').pack(side=Tk.LEFT, fill=Tk.X)
        
        line = ttk.Frame(self); line.pack(fill=Tk.X)
        ttk.Checkbutton(line, text='Classify', variable=self.classify_flag, command=self.classify_flag_changed).pack(side=Tk.LEFT)
        ttk.Button(line, text='Unassign Class', command=self.unassign_class).pack(side=Tk.LEFT)
        
        line = ttk.Frame(self); line.pack(fill=Tk.X)
        ttk.Entry(line, textvariable=self.class_list_fname).pack(side=Tk.LEFT)
        ttk.Button(line, text='Save Classes', command=self.save_class_list).pack(side=Tk.TOP, anchor=Tk.W)
        
        line = ttk.Frame(self); line.pack(fill=Tk.X)
        ttk.Label(line, text='Classification Summary:').pack(side=Tk.TOP, anchor=Tk.W)
        ttk.Label(line, textvariable=self.class_list_summary).pack(side=Tk.TOP, anchor=Tk.W)
        
        self.class_line = ttk.Frame(self); self.class_line.pack(fill=Tk.X)
        self.refresh_class_line()
        
        line = ttk.Frame(self); line.pack(fill=Tk.X)
        ttk.Button(line, text='Prev', command=self.prev_frame).pack(side=Tk.LEFT)
        ttk.Button(line, text='Next', command=self.next_frame).pack(side=Tk.LEFT)
        ttk.Button(line, text='Random', command=self.rand_frame).pack(side=Tk.LEFT)
        ttk.Button(line, text='Refresh', command=self.refresh_class_line).pack(side=Tk.LEFT)
        
    def refresh_class_line(self):
        for c in self.class_line.winfo_children():
            c.destroy()
        ttk.Radiobutton(self.class_line, text='All', variable=self.class_num, value=-1).grid(row=0,column=0)
        for i, k in enumerate(self.classes.key):
            ttk.Radiobutton(self.class_line, text=k, variable=self.class_num, value=i).grid(row=(i+1)/5,column=(i+1)%5)

    def assign_class(self, event=None):
        num = int(self.parent.numstr.get())
        self.classes.clist[num] = event.char
        self.classes.unsaved = True
        self.class_list_summary.set(self.classes.gen_summary())
        self.next_frame()

    def unassign_class(self, event=None):
        num = int(self.parent.numstr.get())
        self.classes.clist[num] = ''
        self.classes.unsaved = True
        self.class_list_summary.set(self.classes.gen_summary())
        self.parent.plot_frame()

    def classify_flag_changed(self, event=None):
        if self.classify_flag.get() == 0:
            for c in string.ascii_lowercase:
                self.parent.master.unbind(c)
        else:
            for c in string.ascii_lowercase:
                self.parent.master.bind(c, self.assign_class)
            self.parent.master.focus()

    def save_class_list(self, event=None):
        print 'Saving manually classified list to', self.class_list_fname.get()
        np.savetxt(self.class_list_fname.get(), self.classes.clist, fmt='%s')
        self.classes.unsaved = False

    def next_frame(self, event=None):
        num = int(self.numstr.get())
        cnum = self.class_num.get()
        if cnum == -1:
            num += 1
        else:
            points = np.where(self.classes.key_pos == cnum)[0]
            index = np.searchsorted(points, num, side='left') + 1
            if index > len(points) - 1:
                index = len(points) - 1
            num = points[index]
        
        if num < self.parent.num_frames:
            self.numstr.set(str(num))
            self.parent.plot_frame()

    def prev_frame(self, event=None):
        num = int(self.numstr.get())
        cnum = self.class_num.get()
        if cnum == -1:
            num -= 1
        else:
            points = np.where(self.classes.key_pos == cnum)[0]
            index = np.searchsorted(points, num, side='left') - 1
            if index < 0:
                index = 0
            num = points[index]
        
        if num > -1:
            self.numstr.set(str(num))
            self.parent.plot_frame()

    def rand_frame(self, event=None):
        cnum = self.class_num.get()
        if cnum == -1:
            num = np.random.randint(self.parent.num_frames)
        else:
            points = np.where(self.classes.key_pos == cnum)[0]
            num = points[np.random.randint(len(points))]
        self.numstr.set(str(num))
        self.parent.plot_frame()

