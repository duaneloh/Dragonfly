import numpy as np
import sys
import os
import string

class Frame_classes():
    def __init__(self, num_frames):
        self.num_frames = num_frames
        self.unsaved = False

    def init_list(self, fname=None):
        if fname is None or not os.path.isfile(fname):
            self.clist = np.zeros((self.num_frames,), dtype=np.str_)
        else:
            print 'Reading class list from', fname
            self.clist = self.read_list(fname)
        self.key, self.key_pos, self.key_counts = np.unique(self.clist, return_inverse=True, return_counts=True)

    def read_list(self, fname):
        with open(fname, 'r') as f:
            c = np.array([l.rstrip() for l in f.readlines()])
        return c

    def gen_summary(self):
        self.key, self.key_pos, self.key_counts = np.unique(self.clist, return_inverse=True, return_counts=True)
        cmin = 0
        summary = ''
        for i in range(len(self.key)):
            summary += '%3s:%-7d' % (self.key[i], self.key_counts[i])
            if i%5 == 4:
                summary += '\n'
        return summary

