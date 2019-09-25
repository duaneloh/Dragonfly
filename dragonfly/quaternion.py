'''Module to generate and manipulate uniformly sampled 3D quaternions'''

from __future__ import print_function
import sys
import numpy as np

NUM_VERT = 120
NUM_EDGE = 720
NUM_FACE = 1200
NUM_CELL = 600
NNN = 12

class QPoints():
    def __init__(self, size):
        self.vec = np.empty((size, 4, 2), dtype='i4')
        self.weight = np.empty((size,), dtype='i4')

class Quaternion():
    def __init__(self, num_div=None):
        self.num_div = num_div
        self.cubic_flag = False
        self.icosahedral_flag = False
        self.num_rot = 0
        self.num_rot_p = 0
        self.quat = None

        if num_div is not None:
            self.generate(num_div)

    def generate(self, num_div):
        self.num_div = num_div

        self._make_600cell()
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

    def _allocate(self):
        self._vertices = np.empty((NUM_VERT, 4))
        self._vec_vertices = np.zeros((NUM_VERT, 4, 2))
        self._edges = np.empty((NUM_EDGE, 2), dtype='i4')
        self._nn_list = [[] for _ in range(NUM_VERT)]
        self._faces = np.empty((NUM_FACE, 3), dtype='i4')
        self._cells = np.empty((NUM_CELL, 4), dtype='i4')
        self._edge2cell = np.empty((NUM_EDGE, 4), dtype='i4')
        self._face2cell = np.empty((NUM_FACE, 4), dtype='i4')

        num = self.num_div
        self.num_rot = 10 * (5 * num**3 + num)
        self.quat = np.empty((self.num_rot, 5))

        self._vertex_points = QPoints(NUM_VERT)
        if num > 1:
            self._edge_points = QPoints(NUM_EDGE * (num - 1))
        if num > 2:
            self._face_points = QPoints(NUM_FACE * (num - 2) * (num - 1) // 2)
        if num > 3:
            self._cell_points = QPoints(NUM_CELL * (num - 3) * (num - 2) * (num - 1) // 6)

    def _make_600cell(self):
        '''Generate vertices, edges, faces and cells for 600-cell sampling of 3-sphere'''
        # Make vertices and vertex weights
        corners = np.stack(np.meshgrid([0., 1.], [0, 1], [0, 1], [0, 1]), -1).reshape(-1, 4)
        self._vertices[:16] = corners - 0.5
        self._vec_vertices[:16,:,0] = (2 * corners - 1) * self.num_div

        self._vertices[16:20] = np.identity(4) * -1
        self._vertices[20:24] = np.identity(4)
        self._vec_vertices[16:20,:,0] = np.identity(4) * -2 * self.num_div
        self._vec_vertices[20:24,:,0] = np.identity(4) * 2 * self.num_div
        
        phi = (np.sqrt(5) - 1 ) / 2.
        vert = np.array([0.5/phi, 0.5, 0.5*phi, 0])
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
        min_dist = phi + 1.e-4

        # Make edges and nearest-neighbour list
        idx = 0
        for i in range(NUM_VERT):
            # Find connected vertices
            sel = np.where(np.linalg.norm(self._vertices[i+1:] - self._vertices[i], axis=1) < min_dist)[0]
            num_sel = sel.shape[0]
            # Add to edge list
            self._edges[idx:idx+num_sel, 0] = i
            self._edges[idx:idx+num_sel, 1] = sel + i + 1
            idx += num_sel

            [self._nn_list[i].append(j+i+1) for j in sel]
            [self._nn_list[j+i+1].append(i) for j in sel]
        self._nn_list = np.array(self._nn_list)

        # Make faces
        idx = 0
        for i, edge in enumerate(self._edges):
            nn = self._nn_list[edge[0]]
            nn = nn[nn > edge[1]]
            test = np.linalg.norm(self._vertices[nn] - self._vertices[edge[1]], axis=1) < min_dist
            sel = np.where(test)[0]
            num_sel = sel.shape[0]
            self._faces[idx:idx+num_sel, 0] = edge[0]
            self._faces[idx:idx+num_sel, 1] = edge[1]
            self._faces[idx:idx+num_sel, 2] = nn[sel]
            idx += num_sel

        # Make cells
        idx = 0
        for i, face in enumerate(self._faces):
            nn = self._nn_list[face[0]]
            nn = nn[nn > face[2]]

            test1 = np.linalg.norm(self._vertices[nn] - self._vertices[face[1]], axis=1) < min_dist
            test2 = np.linalg.norm(self._vertices[nn] - self._vertices[face[2]], axis=1) < min_dist
            sel = np.where(test1 & test2)[0]
            num_sel = sel.shape[0]
            
            self._cells[idx:idx+num_sel, 0] = face[0]
            self._cells[idx:idx+num_sel, 1] = face[1]
            self._cells[idx:idx+num_sel, 2] = face[2]
            self._cells[idx:idx+num_sel, 3] = nn[sel]
            idx += num_sel

        # Make mapping from all edges and faces to one cell each
        for i, edge in enumerate(self._edges):
            nn = self._nn_list[edge[0]]
            nn = nn[nn != edge[1]]
            test = np.linalg.norm(self._vertices[nn] - self._vertices[edge[1]], axis=1) < min_dist
            sel = np.where(test)[0]

            test2 = np.linalg.norm(self._vertices[nn] - self._vertices[nn[sel[0]]], axis=1) < min_dist
            sel2 = np.where(test & test2)[0]
            sel2 = sel2[(nn[sel2]>nn[sel[0]])]

            self._edge2cell[i, 0] = edge[0]
            self._edge2cell[i, 1] = edge[1]
            self._edge2cell[i, 2] = nn[sel[0]]
            self._edge2cell[i, 3] = nn[sel2[0]]
            
        for i, face in enumerate(self._faces):
            nn = self._nn_list[face[0]]
            nn = nn[(nn != face[1]) & (nn != face[2])]

            test1 = np.linalg.norm(self._vertices[nn] - self._vertices[face[1]], axis=1) < min_dist
            test2 = np.linalg.norm(self._vertices[nn] - self._vertices[face[2]], axis=1) < min_dist
            sel = np.where(test1 & test2)[0]
            
            self._face2cell[i, 0] = face[0]
            self._face2cell[i, 1] = face[1]
            self._face2cell[i, 2] = face[2]
            self._face2cell[i, 3] = nn[sel[0]]

    def _quat_setup(self):
        visited_vert = np.zeros(NUM_VERT, dtype='bool')
        
        for i, cell in enumerate(self._cells):
            verts = cell[~visited_vert[cell]]
            if len(verts) == 0:
                continue
            visited_vert[verts] = True
            self._vertex_points.vec[verts] = self._vec_vertices[verts]
            v_q = self._vertices[verts]
            v_c = v_q.sum(0)
            self._vertex_points.weight[verts] = 5.*(v_q*v_c).sum() / (6 * np.linalg.norm(v_q, axis=1)**4 * np.linalg.norm(v_c))

    def _refine_edge(self):
        pass

    def _refine_face(self):
        pass

    def _refine_cell(self):
        pass
