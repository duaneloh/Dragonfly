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
        if self.key[0] == '':
            summary += '|    |%7d|\n' % self.key_counts[0]
            cmin = 1
        for i in range(cmin, len(self.key)):
            summary += '|%-4s|%7d|\n' % (self.key[i], self.key_counts[i])
        return summary

