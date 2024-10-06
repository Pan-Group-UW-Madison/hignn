import os
import hignn
import numpy as np
import sys
from mpi4py import MPI
import time
import h5py

os.system("clear")

if __name__ == '__main__':    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    hignn.Init()
    
    nx = 10
    ny = 10
    nz = 16
    dx = 1.5
    x = np.arange(0, nx * dx, dx)
    y = np.arange(0, ny * dx, dx)
    z = np.arange(0, nz * dx, dx)
    xx, yy, zz = np.meshgrid(x, y, z)
    X = np.concatenate(
        (xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)), axis=1)
    X = X.astype(np.float32)
    
    NN = X.shape[0]

    hignn_model = hignn.HignnModel(X, 100)
    
    hignn_model.load_two_body_model('nn/3D_force_UB_max600_try2')
    
    # set parameters for far dot, the following parameters are default values
    hignn_model.set_epsilon(0.1)
    hignn_model.set_max_iter(15)
    hignn_model.set_mat_pool_size_factor(30)
    hignn_model.set_post_check_flag(False)
    hignn_model.set_use_symmetry_flag(True)
    hignn_model.set_max_far_dot_work_node_size(10000)
    hignn_model.set_max_relative_coord(1000000)
    
    neighbor_list = hignn.NeighborLists()
    neighbor_list.update_coord(X)
    
    neighbor_list.set_two_body_epsilon(5.0)
    neighbor_list.set_three_body_epsilon(5.0)
    
    neighbor_list.build_two_body_info()
    neighbor_list.build_three_body_info()
    
    two_body_edge_info = neighbor_list.get_two_body_edge_info()
    three_body_edge_info = neighbor_list.get_three_body_edge_info()
    
    del hignn_model
    del neighbor_list
    
    hignn.Finalize()