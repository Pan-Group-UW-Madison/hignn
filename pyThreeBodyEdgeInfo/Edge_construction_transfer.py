import numpy as np
from scipy import spatial
from sympy import And
import torch
import copy
import numba as nb
from numba import jit


def create_edge_2body(Nc, force):
    edge_info = np.zeros((2, Nc * Nc))
    a1 = np.arange(0, Nc, 1)
    a1 = np.repeat(a1, axis=0, repeats=Nc)
    a2 = np.arange(0, Nc, 1).reshape(1, Nc)
    a2 = np.repeat(a2, axis=0, repeats=Nc)
    a2 = a2.reshape(1, Nc * Nc)
    edge_info[0, :] = a1
    edge_info[1, :] = a2
    del_ind = np.arange(0, Nc * (Nc + 1), Nc + 1)
    edge_info = np.delete(edge_info, del_ind, axis=1)
    edge_attr = force[edge_info[0, :], :]
    return edge_info, edge_attr


# @jit(nopython=True)
def create_edge_3body(target, Nc, n_dim, R_cut, force, domain, periodic=False):
    target = target.reshape(Nc, n_dim)
    if periodic == False:
        source = target
        source_index = np.arange(0, Nc, dtype='int32')
    else:
        # build ghost coords and index for periodic domain
        core_domain = np.copy(domain)
        core_domain[0, :] += np.ones((n_dim), dtype='float32') * R_cut
        core_domain[1, :] -= np.ones((n_dim), dtype='float32') * R_cut

        domain_size = domain[1, :] - domain[0, :]

        num_source_duplicate = np.zeros((Nc), dtype='int32')
        axis_source_duplicate = np.ones((Nc, n_dim), dtype='int32')
        num_source_duplicate[0] = 0
        for i in range(Nc):
            num = 1
            for j in range(n_dim):
                if target[i, j] < core_domain[0, j]:
                    axis_source_duplicate[i, j] = -2
                    num *= 2
                elif target[i, j] > core_domain[1, j]:
                    axis_source_duplicate[i, j] = 2
                    num *= 2
                else:
                    axis_source_duplicate[i, j] = 1

            num_source_duplicate[i] = num

        num_source_offset = np.zeros((Nc+1), dtype='int32')
        num_source_offset[1:None] = num_source_duplicate
        num_source_offset[0] = 0
        num_source_offset = np.cumsum(num_source_offset)
        source = np.zeros((num_source_offset[-1], n_dim), dtype='float32')
        source_index = np.zeros((num_source_offset[-1]), dtype='int32')
        for i in range(Nc):
            offset = np.zeros(
                (num_source_duplicate[i], n_dim), dtype='float32')

            num = num_source_duplicate[i]
            stride1 = num_source_duplicate[i]
            for j in range(n_dim):
                if axis_source_duplicate[i, j] == 1:
                    offset[:, j] = 0
                if axis_source_duplicate[i, j] == 2:
                    for m in range(0, num, stride1):
                        offset[np.arange(m, m+stride1/2, dtype='int32'), j] = 0
                        offset[np.arange(m+stride1/2, m+stride1,
                                         dtype='int32'), j] = -domain_size[j]
                    stride1 = int(stride1 / 2)
                if axis_source_duplicate[i, j] == -2:
                    for m in range(0, num, stride1):
                        offset[np.arange(m, m+stride1/2, dtype='int32'), j] = 0
                        offset[np.arange(m+stride1/2, m+stride1,
                                         dtype='int32'), j] = domain_size[j]
                    stride1 = int(stride1 / 2)

            source[num_source_offset[i]:num_source_offset[i+1],
                   :] = np.tile(target[i], (num, 1)) + offset
            source_index[num_source_offset[i]:num_source_offset[i+1]] = i

    tree = spatial.cKDTree(source)
    edge_info = np.zeros((Nc * Nc * 20, 3), dtype=int)
    edge_info_self = np.zeros((Nc * Nc, 2), dtype=int)
    k1 = 0
    k2 = 0
    edge = tree.query_ball_point(target, R_cut)

    # reorganize edge
    if periodic == True:
        for i in range(Nc):
            edge[i] = source_index[list(edge[i])].tolist()

    for i in range(Nc):
        edge1 = list(edge[i])
        edge1.remove(i)
        edge_info_self[k2:(k2 + len(edge1)), 0] = i
        edge_info_self[k2:(k2 + len(edge1)), 1] = edge1
        k2 += len(edge1)
        # edge1_set = set(edge1)
        for j in edge1:
            # edge2 = np.intersect1d(edge[j], edge1, assume_unique=True)
            # edge2 = list(edge1_set.intersection(edge[j]))
            # edge2 = source_index[list(edge[i])].tolist()
            edge2 = list(edge[j])
            edge2.remove(j)
            edge2.remove(i)
            edge_info[k1:(k1 + len(edge2)), 0] = i
            edge_info[k1:(k1 + len(edge2)), 1] = j
            edge_info[k1:(k1 + len(edge2)), 2] = edge2
            k1 += len(edge2)
    edge_info = edge_info[0:k1, :].T
    edge_info_self = edge_info_self[0:k2, :].T
    edge_attr = torch.cat((force[edge_info[0, :]], force[edge_info[2, :]]), 1)
    edge_attr_self = force[edge_info_self[1, :], :]
    return edge_info, edge_attr, edge_info_self, edge_attr_self
