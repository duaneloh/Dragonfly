import numpy as np
import sys
import os
import string

class Frame_classes():
    def __init__(self, num_frames, fname='my_classes.dat'):
        self.num_frames = num_frames
        self.unsaved = False
        self.fname = fname
        self.init_list()

    def init_list(self):
        if self.fname is '' or not os.path.isfile(self.fname):
            sys.stderr.write('Initializing empty class list\n')
            self.clist = np.zeros((self.num_frames,), dtype=np.str_)
        else:
            self.clist = self.load()
        self.key, self.key_pos, self.key_counts = np.unique(self.clist, return_inverse=True, return_counts=True)

    def load(self):
        with open(os.path.realpath(self.fname), 'r') as f:
            c = np.array([l.rstrip() for l in f.readlines()])
            c[c==''] = ' '
        sys.stderr.write('Read class list from %s\n' % self.fname)
        self.unsaved = False
        return c

    def save(self):
        sys.stderr.write('Saving manually classified list to %s\n' % self.fname)
        np.savetxt(os.path.realpath(self.fname), self.clist, fmt='%s')
        self.unsaved = False

    def gen_summary(self):
        self.key, self.key_pos, self.key_counts = np.unique(self.clist, return_inverse=True, return_counts=True)
        cmin = 0
        summary = ''
        for i in range(len(self.key)):
            summary += '%3s:%-7d' % (self.key[i], self.key_counts[i])
            if i%5 == 4:
                summary += '\n'
        return summary

