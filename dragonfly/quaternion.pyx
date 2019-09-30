'''Module to generate and manipulate uniformly sampled 3D quaternions'''

from __future__ import print_function
import sys
import os
from socket import gethostname
import itertools
import numpy as np
import pandas
from scipy.spatial import distance

NUM_VERT = 120
NUM_EDGE = 720
NUM_FACE = 1200
NUM_CELL = 600
NNN = 12

class Quaternion():
    def __init__(self, num_div=None, point_group=''):
        self.num_div = num_div
        self.cubic_flag = (point_group == 'S4')
        self.icosahedral_flag = (point_group == 'A5')
        self.num_rot = 0
        self.num_rot_p = 0
        self.quat = None

        self._reduced = False
        self._vertices = None
        self._edges = None
        self._faces = None
        self._cells = None
        self._vec_vertices = None
        self._nn_list = None
        self._edge2cell = None
        self._face2cell = None

        if num_div is not None:
            self.generate(num_div)

    def generate(self, num_div):
        self.num_div = num_div
        self.num_rot = 10 * (5 * num_div**3 + num_div)
        self.quat = np.empty((self.num_rot, 5))

        self._get_600cell()

        offset = self._vertex_quats()
        if num_div > 1:
            offset = self._edge_quats(offset)
        if num_div > 2:
            offset = self._face_quats(offset)
        if num_div > 3:
            offset = self._cell_quats(offset)

        if self.icosahedral_flag:
            self.reduce_icosahedral()
        elif self.cubic_flag:
            self.reduce_cubic()

        self.quat[:,4] /= self.quat[:,4].sum()

        return self.num_rot

    def save(self, fname):
        if self.quat is None:
            raise AttributeError('Generate quaternion first before saving')

        with open(fname, 'w') as fptr:
            fptr.write('%d\n' % self.num_rot)
            np.savetxt(fptr, self.quat, fmt='%+17.15f')

    def parse(self, fname):
        df = pandas.read_csv(fname, sep='\s+', skiprows=1, header=None, dtype='f8')
        self.quat = df.values
        self.num_rot = self.quat.shape[0]

        if self.quat.shape[1] == 5:
            self.quat[:,4] /= self.quat[:,4].sum()
        elif self.quat.shape[1] == 4:
            self.quat = np.pad(self.quat, ((0,0),(0,1)),
                               mode='constant', constant_values=1./self.num_rot)
        else:
            raise ValueError('Unknown shape for data in %s'%fname)

    def divide(self, rank, num_proc, num_modes=1, num_nonrot_modes=0):
        tot_num_rot = num_modes * self.num_rot + num_nonrot_modes
        self.num_rot_p = tot_num_rot / num_proc
        if rank < (tot_num_rot % num_proc):
            self.num_rot_p += 1
        if num_proc > 1:
            print("%d: %s: num_rot_p = %d/%d\n", rank, gethostname(), self.num_rot_p, tot_num_rot)

    def reduce_icosahedral(self):
        if self.quat is None:
            raise AttributeError('Generate quaternion first before reducing')
        if self._reduced:
            print('Already reduced')
            return self.num_rot

        # Icosahedral point group quaternions are just the vertex ones
        # Keep quaternions whose closest vertex_quat is the identity
        vert_dist = distance.cdist(self.quat[:60,:4], self.quat[60:,:4])
        sel = np.where(vert_dist.argmin(axis=0) == 8)[0]
        self.num_rot = 1 + sel.shape[0]
        self.quat[0] = np.copy(self.quat[8])
        self.quat[1:self.num_rot] = self.quat[sel+60]
        self.quat = np.copy(self.quat[:self.num_rot])

        self._reduced = True
        return self.num_rot

    def reduce_cubic(self):
        if self.quat is None:
            raise AttributeError('Generate quaternion first before reducing')
        if self._reduced:
            print('Already reduced')
            return self.num_rot

        # Generate cubic point group quaternions
        cube_quat = np.zeros((24, 4))

        # -- neut and inv2 (1x + 3x)
        cube_quat[:4] = np.identity(4)

        # -- rot3 (8x)
        signs3 = np.stack(np.meshgrid([-1., 1.], [-1., 1.], [-1., 1.], indexing='ij'), -1).reshape(-1, 3)
        cube_quat[4:12] = 0.5
        cube_quat[4:12,1:] *= signs3

        # -- rot1 (6x)
        val = np.sqrt(0.5)
        cube_quat[12:18, 0] = val
        cube_quat[12:15, 1:] = np.identity(3) * val
        cube_quat[15:18, 1:] = np.identity(3) * val * -1

        # -- rot2 (6x)
        for i, pos in enumerate(itertools.permutations(range(3), 2)):
            cube_quat[18 + i, np.array(pos) + 1] = [val, -val] if pos[0] < pos[1] else [val, val]

        # Keep quaternions whose closest cube_quat is the identity
        cube_dist = distance.cdist(cube_quat, self.quat[:,:4])
        sel = np.where(cube_dist.argmin(axis=0) == 0)[0]
        self.num_rot = sel.shape[0]
        self.quat[:self.num_rot] = self.quat[sel]
        self.quat = np.copy(self.quat[:self.num_rot])

        self._reduced = True
        return self.num_rot

    def _get_600cell(self):
        fname = os.path.dirname(__file__)+'/../aux/600cell.npz'
        if os.path.isfile(fname):
            data = np.load(fname)
            self._vertices = data['vertices']
            self._edges = data['edges']
            self._faces = data['faces']
            self._cells = data['cells']
            self._vec_vertices = data['vec_vertices']
            self._nn_list = data['nn_list']
            self._edge2cell = data['edge2cell']
            self._face2cell = data['face2cell']
        else:
            self._make_600cell()
            self._save_600cell()
        self._vec_vertices *= self.num_div

    def _make_600cell(self):
        '''Generate vertices, edges, faces and cells for 600-cell sampling of 3-sphere'''
        self._vertices = np.empty((NUM_VERT, 4))
        self._vec_vertices = np.zeros((NUM_VERT, 4, 2), dtype='i4')
        self._edges = np.empty((NUM_EDGE, 2), dtype='i4')
        self._nn_list = [[] for _ in range(NUM_VERT)]
        self._faces = np.empty((NUM_FACE, 3), dtype='i4')
        self._cells = np.empty((NUM_CELL, 4), dtype='i4')
        self._edge2cell = np.empty((NUM_EDGE, 4), dtype='i4')
        self._face2cell = np.empty((NUM_FACE, 4), dtype='i4')

        # Make vertices and vertex weights
        corners = np.stack(np.meshgrid([0., 1.], [0, 1], [0, 1], [0, 1], indexing='ij'), -1).reshape(-1, 4)
        self._vertices[:16] = corners - 0.5
        self._vec_vertices[:16,:,0] = (2 * corners - 1)

        self._vertices[16:20] = np.identity(4) * -1
        self._vertices[20:24] = np.identity(4)
        self._vec_vertices[16:20,:,0] = np.identity(4) * -2
        self._vec_vertices[20:24,:,0] = np.identity(4) * 2

        phi = (np.sqrt(5) - 1 ) / 2.
        vert = np.array([0.5/phi, 0.5, 0.5*phi, 0])
        vec_vert = np.zeros((4, 2), dtype='i4')
        vec_vert[0,1] = 1
        vec_vert[1,0] = 1
        vec_vert[2,0] = -1
        vec_vert[2,1] = 1
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
        # -- Get pairwise distances
        dist = distance.squareform(distance.pdist(self._vertices))
        for i in range(NUM_VERT):
            # Find connected vertices
            sel = np.where(dist[i,i+1:] < min_dist)[0]
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
            sel2 = sel2[nn[sel2] > nn[sel[0]]]

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

    def _save_600cell(self):
        out_fname = os.path.dirname(__file__)+'/../aux/600cell.npz'
        out_dict = {
                'vertices': self._vertices,
                'edges': self._edges,
                'faces': self._faces,
                'cells': self._cells,
                'vec_vertices': self._vec_vertices,
                'nn_list': self._nn_list,
                'edge2cell': self._edge2cell,
                'face2cell': self._face2cell}
        np.savez(out_fname, **out_dict)

    def _vertex_quats(self):
        visited_vert = np.zeros(NUM_VERT, dtype='bool')
        tau = (np.sqrt(5) + 1) / 2.
        num_quat = 1
        vecs = np.zeros((num_quat*NUM_VERT, 4, 2), dtype='i4')
        weights = np.zeros(num_quat*NUM_VERT)

        for i, cell in enumerate(self._cells):
            verts = cell[~visited_vert[cell]]
            if verts.size == 0:
                continue

            v_q = self._vertices[cell]
            v_c = v_q.sum(0)
            w = (v_q*v_c).sum(1) / np.linalg.norm(v_q, axis=1)**4 / np.linalg.norm(v_c)

            vecs[verts] = self._vec_vertices[verts]
            weights[verts] = 5 * w[~visited_vert[cell]] / 6
            visited_vert[verts] = True

        vsign = np.sign(vecs).reshape(-1, 8)
        sel = np.array([vs[vs!=0][0] > 0 for vs in vsign])
        vquats = (vecs[sel,:,0] + vecs[sel,:,1]*tau) / 2 / self.num_div
        end = num_quat*NUM_VERT//2
        self.quat[:end, :4] = vquats / np.linalg.norm(vquats, axis=1, keepdims=True)
        self.quat[:end, 4] = weights[sel]

        return end

    def _edge_quats(self, offset):
        num = self.num_div
        tau = (np.sqrt(5) + 1) / 2.
        num_quat = num - 1
        vecs = np.zeros((num_quat*NUM_EDGE, 4, 2), dtype='i4')
        weights = np.zeros(num_quat*NUM_EDGE)

        for i, edge in enumerate(self._edges):
            start_v = self._vec_vertices[edge[0]]
            end_v = self._vec_vertices[edge[1]]
            vec_d_v = (end_v - start_v) / num

            v_c = self._vertices[self._edge2cell[i]].sum(0)
            v_interp = np.array([start_v + j * vec_d_v
                                 for j in np.arange(1, num)])
            v_q = (v_interp[:,:,0] + v_interp[:,:,1]*tau) / 2 / num
            w = (v_q*v_c).sum(1) / np.linalg.norm(v_q, axis=1)**4 / np.linalg.norm(v_c)

            vecs[i*num_quat:(i+1)*num_quat] = v_interp
            weights[i*num_quat:(i+1)*num_quat] = 35 * w / 36

        vsign = np.sign(vecs).reshape(-1, 8)
        sel = np.array([vs[vs!=0][0] > 0 for vs in vsign])
        vquats = (vecs[sel,:,0] + vecs[sel,:,1]*tau) / 2 / self.num_div
        end = offset + num_quat * NUM_EDGE // 2
        self.quat[offset:end, :4] = vquats / np.linalg.norm(vquats, axis=1, keepdims=True)
        self.quat[offset:end, 4] = weights[sel]

        return end

    def _face_quats(self, offset):
        num = self.num_div
        tau = (np.sqrt(5) + 1) / 2.
        num_quat = (num - 1) * (num - 2) // 2
        vecs = np.zeros((num_quat*NUM_FACE, 4, 2), dtype='i4')
        weights = np.zeros(num_quat*NUM_FACE)

        for i, face in enumerate(self._faces):
            start_v = self._vec_vertices[face[0]]
            vec_d_v1 = (self._vec_vertices[face[1]] - start_v) / num
            vec_d_v2 = (self._vec_vertices[face[2]] - start_v) / num

            v_c = self._vertices[self._face2cell[i]].sum(0)
            v_interp = np.array([start_v + j*vec_d_v1 + k*vec_d_v2
                                 for j in range(1, num-1)
                                 for k in range(1, num-j)])
            v_q = (v_interp[:,:,0] + v_interp[:,:,1]*tau) / 2 / num
            w = (v_q*v_c).sum(1) / np.linalg.norm(v_q, axis=1)**4 / np.linalg.norm(v_c)

            vecs[i*num_quat:(i+1)*num_quat] = v_interp
            weights[i*num_quat:(i+1)*num_quat] = w

        vsign = np.sign(vecs).reshape(-1, 8)
        sel = np.array([vs[vs!=0][0] > 0 for vs in vsign])
        vquats = (vecs[sel,:,0] + vecs[sel,:,1]*tau) / 2 / self.num_div
        end = offset + num_quat * NUM_FACE // 2
        self.quat[offset:end, :4] = vquats / np.linalg.norm(vquats, axis=1, keepdims=True)
        self.quat[offset:end, 4] = weights[sel]

        return end

    def _cell_quats(self, offset):
        num = self.num_div
        tau = (np.sqrt(5) + 1) / 2.
        num_quat = (num - 1) * (num - 2) * (num - 3) // 6
        vecs = np.zeros((num_quat*NUM_CELL, 4, 2), dtype='i4')
        weights = np.zeros(num_quat*NUM_CELL)

        for i, cell in enumerate(self._cells):
            start_v = self._vec_vertices[cell[0]]
            vec_d_v1 = (self._vec_vertices[cell[1]] - start_v) / num
            vec_d_v2 = (self._vec_vertices[cell[2]] - start_v) / num
            vec_d_v3 = (self._vec_vertices[cell[3]] - start_v) / num

            v_c = self._vertices[cell].sum(0)
            v_interp = np.array([start_v + j*vec_d_v1 + k*vec_d_v2 + m*vec_d_v3
                                 for j in range(1,num-2)
                                 for k in range(1,num-1-j)
                                 for m in range(1,num-j-k)])
            v_q = (v_interp[:,:,0] + v_interp[:,:,1]*tau) / 2 / num
            w = (v_q*v_c).sum(1) / np.linalg.norm(v_q, axis=1)**4 / np.linalg.norm(v_c)

            vecs[i*num_quat:(i+1)*num_quat] = v_interp
            weights[i*num_quat:(i+1)*num_quat] = w

        vsign = np.sign(vecs).reshape(-1, 8)
        sel = np.array([vs[vs!=0][0] > 0 for vs in vsign])
        vquats = (vecs[sel,:,0] + vecs[sel,:,1]*tau) / 2 / self.num_div
        end = offset + num_quat * NUM_CELL // 2
        self.quat[offset:end, :4] = vquats / np.linalg.norm(vquats, axis=1, keepdims=True)
        self.quat[offset:end, 4] = weights[sel]

        return end

    @staticmethod
    def qdist(q1, q2):
        return 1. - (q1*q2).sum(-1)**2

