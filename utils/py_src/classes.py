'''Module to handle frame-by-frame classification'''

from __future__ import print_function
import sys
import os
import numpy as np

class FrameClasses(object):
    '''Container class for frame classes
    Attributes:
        num_frames: Number of frames
        unsaved: Flag on whether the current class list has been saved
        fname: File name to save/load
        clist: Class list
    Methods:
        load(): Load class list from file
        save(): Save class list to file
        gen_summary(): Generate a summary string for class list
    '''
    def __init__(self, num_frames, fname=None):
        self.num_frames = num_frames
        self.unsaved = False
        if fname is None:
            self.fname = 'my_classes.dat'
        else:
            self.fname = fname

        if self.fname == '' or not os.path.isfile(self.fname):
            sys.stderr.write('Initializing empty class list\n')
            self.clist = np.zeros((self.num_frames,), dtype='U')
        else:
            self.clist = self.load()
        self.key, self.key_pos, self.key_counts = np.unique(
            self.clist, return_inverse=True, return_counts=True)

    def load(self):
        '''Load class list from self.fname'''
        with open(os.path.realpath(self.fname), 'r') as fptr:
            classes = np.array([l.rstrip() for l in fptr.readlines()])
            classes[classes == ''] = ' '
        sys.stderr.write('Read class list from %s\n' % self.fname)
        self.unsaved = False
        return classes

    def save(self):
        '''Save self.clist to self.fname'''
        sys.stderr.write('Saving manually classified list to %s\n' % self.fname)
        np.savetxt(os.path.realpath(self.fname), self.clist, fmt='%s')
        self.unsaved = False

    def gen_summary(self):
        '''Generate summary string from self.clist'''
        self.key, self.key_pos, self.key_counts = np.unique(
            self.clist, return_inverse=True, return_counts=True)
        summary = ''
        for i in range(len(self.key)):
            summary += '%3s:%-7d' % (self.key[i], self.key_counts[i])
            if i%5 == 4:
                summary += '\n'
        return summary
