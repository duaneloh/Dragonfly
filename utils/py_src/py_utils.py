import numpy as np
import time

class my_timer(object):
    def __init__(self):
        self.t0 = time.time()
        self.ts = self.t0

    def reset(self):
        t1 = time.time()
        self.t0 = t1

    def reset_and_report(self, msg):
        t1 = time.time()
        print "{:-<30}:{:5.5f} seconds".format(msg, t1-self.t0)
        self.t0 = t1

    def report_time_since_beginning(self):
        print "="*80
        print "{:-<30}:{:5.5f} seconds".format("Since beginning", time.time() - self.ts)


def write_density(in_den_file, in_den, binary=True):
    if binary:
        in_den.astype('float64').tofile(in_den_file)
    else:
        with open(in_den_file, "w") as fp:
            for l0 in in_den:
                for l1 in l0:
                    tmp = ' '.join(l1.astype('str'))
                    fp.write(tmp + '\n')

def read_density(in_den_file, binary=True):
    if binary:
        den     = np.fromfile(in_den_file, dtype="float64")
        sz      = len(den)
        l       = int(np.round(np.power(sz, 1./3.)))
        out_den = den.reshape(l,l,l)
    else:
        with open(in_den_file, "r") as fp:
            lines   = fp.readlines()
            den     = []
            for l in lines:
                den.append([float(s) for s in l.strip().split()])
            den     = np.array(den)
        sz      = len(den)
        l       = int(np.power(sz, 1./3.))
        out_den = den.reshape(l,l,l)
    return out_den
