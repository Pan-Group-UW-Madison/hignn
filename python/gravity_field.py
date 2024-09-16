import os
import hignn
import numpy as np
import sys
from mpi4py import MPI
import time
import h5py

os.system("clear")

def velocity_update(t, position):
    # update coordinates
    hignn_model.update_coord(position[:, 0:3])
    
    # update force/potential
    force = np.zeros((position.shape[0], 3), dtype=np.float32)
    force[:, 2] = -1.0
    
    # create velocity array
    velocity = np.zeros((position.shape[0], 3), dtype=np.float32)
    
    hignn_model.dot(velocity, force)
    
    return velocity

if __name__ == '__main__':
    # initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    hignn.Init()
    
    X = np.loadtxt('output0.txt', dtype=np.float32)
    
    NN = X.shape[0]

    hignn_model = hignn.HignnModel(X, 50)
    
    hignn_model.load_two_body_model('nn/3D_force_UB_max600_try2')
    
    # set parameters for far dot, the following parameters are default values
    hignn_model.set_epsilon(0.1)
    hignn_model.set_max_iter(15)
    hignn_model.set_mat_pool_size_factor(30)
    hignn_model.set_post_check_flag(False)
    hignn_model.set_use_symmetry_flag(True)
    hignn_model.set_max_far_dot_work_node_size(10000)
    hignn_model.set_max_relative_coord(1000000)
    
    # setup time integrator
    time_integrator = hignn.ExplicitEuler()
    
    time_integrator.set_time_step(0.005)
    time_integrator.set_final_time(0.01)
    time_integrator.set_num_rigid_body(NN)
    time_integrator.set_output_step(1)
    
    time_integrator.set_velocity_func(velocity_update)
    time_integrator.initialize(X)
    
    t1 = time.time()
    
    time_integrator.run()
    
    del hignn_model
    del time_integrator
    
    hignn.Finalize()