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

