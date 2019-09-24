'''Module to generate and manipulate uniformly sampled 3D quaternions'''

from __future__ import print_function
import sys
import numpy as np

NUM_VERT = 120
NUM_EDGE = 720
NUM_FACE = 1200
NUM_CELL = 600
NNN = 12

class Quaternion():
    def __init__(self, num_div=None):
        self.num_div = num_div
        self.cubic_flag = False
        self.icosahedral_flag = False
        self.num_rot = 0
        self.num_rot_p = 0

        if num_div is not None:
            self.generate(num_div)

    def generate(self, num_div):
        self.num_div = num_div

        self._make_300cell()
        self._quat_setup()

        if num_div > 1:
            self._refine_edge()
        if num_div > 2:
            self._refine_face()
        if num_div > 3:
            self._refine_cell()

        self._print_quat()

        if self.icosahedral_flag:
            self.reduce_icosahedral()
        elif self.cubic_flag:
            self.reduce_cubic()

        self.quat[:,4] /= self.quat[:,4].sum()

        return self.num_rot

    def _make_300cell(self):
        '''
        min_dist2 = self.calc_min_dist2()
        self._make_edge(min_dist2)
        self._make_face(min_dist2)
        self._make_cell(min_dist2)
        self._make_map(min_dist2)
        '''

        # Make vertices and vertex weights
        self._vertices = np.empty((NUM_VERT, 4))
        self._vec_vertices = np.zeros((NUM_VERT, 4, 2))

        corners = np.stack(np.meshgrid([0., 1.], [0, 1], [0, 1], [0, 1]), -1).reshape(-1, 4)
        self._vertices[:16] = corners - 0.5
        self._vec_vertices[:16,:,0] = (2 * corners - 1) * self.num_div

        self._vertices[16:20] = np.identity(4) * -1
        self._vertices[20:24] = np.identity(4)
        self._vec_vertices[16:20,:,0] = np.identity(4) * -2 * self.num_div
        self._vec_vertices[20:24,:,0] = np.identity(4) * 2 * self.num_div
        
        tau = (np.sqrt(5) + 1 ) / 2.
        vert = np.array([0.5*tau, 0.5, 0.5/tau, 0])
        vec_vert = np.zeros((4, 2))
        vec_vert[0,1] = self.num_div
        vec_vert[1,0] = self.num_div
        vec_vert[2,0] = -self.num_div
        vec_vert[2,1] = self.num_div
        corners[:,-1] = 0.5
        signs = 2 * np.unique(corners, axis=0) - 1
        perms = np.array([[0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2],
                          [1, 2, 0, 3], [1, 0, 3, 2], [1, 3, 2, 0],
                          [2, 0, 1, 3], [2, 3, 0, 1], [2, 1, 3, 0],
                          [3, 1, 0, 2], [3, 0, 2, 1], [3, 2, 1, 0]])
        self._vertices[24:] = [(vert*s)[p] for s in signs for p in perms]
        self._vec_vertices[24:,:,0] = [(vec_vert[:,0]*s)[p] for s in signs for p in perms]
        self._vec_vertices[24:,:,1] = [(vec_vert[:,1]*s)[p] for s in signs for p in perms]

        # Edge length
        min_dist2 = 1. / tau + 1.e-4

        # Make edges and nearest-neighbour list
        self._edges = np.empty((NUM_EDGE, 2), dtype='i4')
        self._nn_list = [[] for _ in range(NUM_VERT)]
        idx = 0
        for i in range(NUM_VERT):
            # Find connected vertices
            sel = np.where(np.linalg.norm(self._vertices[i+1:] - self._vertices[i], axis=1) < min_dist2)[0]
            num_sel = sel.shape[0]
            # Add to edge list
            self._edges[idx:idx+num_sel, 0] = i
            self._edges[idx:idx+num_sel, 1] = sel + i + 1
            idx += num_sel
            [self._nn_list[i].append(j+i+1) for j in sel]
            [self._nn_list[j+i+1].append(i) for j in sel]
        self._nn_list = np.array(self._nn_list)

        # Make faces
        # Make cells
