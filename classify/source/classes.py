import numpy as np
import sys
import os
import string

class Frame_classes():
    def __init__(self, num_frames):
        self.num_frames = num_frames
        self.unsaved = False

    def init_list(self, fname=None):
        if fname is None:
            self.clist = np.zeros((self.num_frames,), dtype=np.str_)
        else:
            print 'Reading class list from', fname
            self.clist = self.read_list(fname)

    def read_list(self, fname):
        with open(fname, 'r') as f:
            c = np.array([l.rstrip() for l in f.readlines()])
        self.key, self.key_pos = np.unique(c, return_inverse=True)
        return c

    def gen_summary(self):
        u = np.unique(self.clist, return_counts=True)
        cmin = 0
        summary = ''
        if u[0][0] == '':
            summary += '|    |%7d|\n' % u[1][0]
            cmin = 1
        for i in range(cmin, len(u[0])):
            summary += '|%-4s|%7d|\n' % (u[0][i], u[1][i])
        return summary

