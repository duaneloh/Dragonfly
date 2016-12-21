import numpy as np
import sys

class Polar_converter():
    def __init__(self, x, y, mask, r_min=5, r_max=60, delta_r=2., delta_ang=10.):
        self.x = x.flatten()
        self.y = y.flatten()
        self.mask = mask.flatten()
        self.first_pass = True
        self.delta_r = delta_r
        self.delta_ang = delta_ang
        self.r_min = int(r_min/delta_r)
        self.r_max = int(np.ceil(r_max/delta_r))

    def convert(self, input_frame):
        """
        Converts each input diffraction pattern into a polar representation
        """
        if self.first_pass:
            #Compute the angular and radial bins using the first encountered frame.
            ang = np.mod(np.arctan2(self.y, self.x), np.pi) / (np.pi*self.delta_ang/180.)
            self.angs = (ang - ang.min()).astype('i4')
            self.rads = (np.sqrt(self.x*self.x + self.y*self.y)/self.delta_r).astype('i4')
            (self.ang_max, self.rad_max,) = (self.angs.max()+1, self.rads.max()+1)
            self.polar_indices = [a*self.ang_max+b for a,b in zip(self.rads, self.angs)]
            
            self.polar_count = np.zeros(self.rad_max*self.ang_max)
            np.add.at(self.polar_count, self.polar_indices, self.mask)
            self.first_pass = False
        
        self.polar_arr = np.zeros(self.rad_max*self.ang_max)
        np.add.at(self.polar_arr, self.polar_indices, input_frame.flatten()*self.mask)
        self.polar_arr = (self.polar_arr/(1.*(self.polar_count + (self.polar_count==0)))).reshape(self.rad_max, -1)
        
        return self.polar_arr

    def compute_ang_corr(self, ang_max=10):
        """
        Compute the angular correlation from the polar representation of each pattern
        """
        self.ang_corr = np.array([a - a.mean() for a in self.polar_arr[self.r_min:self.r_max]])
        temp = []
        for a in self.ang_corr:
            la = np.linalg.norm(a)
            if la > 0.:
                temp.append(np.absolute(np.fft.fft(a/la))[1:ang_max])
            else:
                temp.append(np.zeros(ang_max-1))
        self.ang_corr = np.array(temp)
        
        return self.ang_corr

