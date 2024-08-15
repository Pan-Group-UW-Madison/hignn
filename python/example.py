import os
import hignn
import numpy as np
import sys
from mpi4py import MPI
import time
import h5py

os.system("clear")

def velocity_update(t, position):
    if rank == 0:
        print("t = {t:.4f}".format(t = t))
    hignn_model.update_coord(position[:, 0:3])
    velocity = np.zeros((position.shape[0], 3), dtype=np.float32)
    force = np.zeros((position.shape[0], 3), dtype=np.float32)
    force[:, 2] = -1.0
    
    hignn_model.dot(velocity, force)
    
    return velocity

if __name__ == '__main__':    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    hignn.Init()
    
    N = 5
    nx = N
    ny = N
    nz = N
    dx = 3
    x = np.arange(0, nx * dx, dx)
    y = np.arange(0, ny * dx, dx)
    z = np.arange(0, nz * dx, dx)
    xx, yy, zz = np.meshgrid(x, y, z)
    X = np.concatenate(
        (xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)), axis=1)
    X = X.astype(np.float32)
    
    # X = np.loadtxt('output0.txt', dtype=np.float32)
    # X = np.loadtxt('output23.txt', dtype=np.float32)
    # X = np.loadtxt('Result/pos2.txt', dtype=np.float32)
    
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
    
    # time_integrator = hignn.ExplicitEuler()
    
    # time_integrator.set_time_step(0.0001)
    # time_integrator.set_final_time(0.0002)
    # time_integrator.set_num_rigid_body(NN)
    # time_integrator.set_output_step(1)
    
    # time_integrator.set_velocity_func(velocity_update)
    # time_integrator.initialize(X)
    # time_integrator.run()
    
    rank_range = np.linspace(0, NN, comm.Get_size() + 1, dtype=np.int32)
    
    # for i in range(10):
    #     if rank == 0:
    #         print(i)
    #     v = velocity_update(0, X)
    ts = 0
    dt = 0.005
    ite = 0
    
    t1 = time.time()

    # explicit euler time integrator
    # equivalent to line 69
    for i in range(2):
        with h5py.File('Result/pos'+str(ite)+'rank'+str(rank)+'.h5', 'w') as f:
            f.create_dataset('pos', data=X[rank_range[rank]:rank_range[rank+1], :])
        
        tt1 = time.time()
        V = velocity_update(ts, X)
        if rank == 0:
            print("Time for velocity_update: {t:.4f}s".format(t = time.time() - tt1))

        with h5py.File('Result/vel'+str(ite)+'rank'+str(rank)+'.h5', 'w') as f:
            f.create_dataset('vel', data=V[rank_range[rank]:rank_range[rank+1], :])
        
        X = X + dt * V
        ts = ts + dt
        
        ite = ite + 1
        if rank == 0:
            print()            

    # if rank == 0:
    #     print("Time for simulation: {t:.4f}s".format(t = time.time() - t1))
   
    # edgeInfo = hignn.BodyEdgeInfo()
    # edgeInfo.setThreeBodyEpsilon(5.0)

    # edgeInfo.setTargetSites(X)

    # edgeInfo.buildThreeBodyEdgeInfo()
    
    del hignn_model
    # del edgeInfo
    # del time_integrator
    
    hignn.Finalize()