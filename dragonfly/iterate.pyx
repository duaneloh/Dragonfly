'''Current: Dumping ground for iterate functions from other classes'''

class Iterate():
    def parse_scale(self, fname, bg=False):
        if h5py.is_hdf5(fname):
            with h5py.File(fname, 'r') as f:
                scale = f['scale'][:]
        else:
            scale = pandas.read_csv(fname, header=None).array.ravel()
        
        if bg:
            self.bgscale = scale
        else:
            self.scale = scale

    def normalize_scale(self, frames):
        blist = frames.blacklist
        mean_scale = self.scale[blist==0].mean()
        self.model1 * mean_scale
        self.scale[blist==0] /= mean_scale
        self.rms_change *= mean_scale

    def parse_blacklist(self, fname, sel_string=None):
        '''Generate blacklist from file and selection string
        
        Blacklist file contains one number (0 or 1) per line for each frame indicating whether
        the frame is blacklisted (1) or considered good (0).
        
        On top of that for dataset splitting, one can provide a selection string, either
        'odd_only' or 'even_only' to take only half of the good frames.
        '''
        cdef uint8_t[:] arr
        if os.path.isfile(fname):
            arr = pandas.read_csv(fname, header=None, squeeze=True, dtype='u1').array
            self.dset.blacklist = <uint8_t*> malloc(arr.shape[0] * sizeof(uint8_t))
            memcpy(&self.dset.blacklist, &arr[0], arr.shape[0])

        if sel_string is 'odd_only':
            self.blacklist[self.blacklist==0][0::2] = 1
        elif sel_string is 'even_only':
            self.blacklist[self.blacklist==0][1::2] = 1

    @staticmethod
    def calculate_size(qmax):
        return 2 * np.ceil(qmax) + 3

