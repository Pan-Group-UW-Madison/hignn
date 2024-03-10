import os
import hignn
import numpy as np
import sys
from mpi4py import MPI
import time

def velocity_update(t, position):
    if rank == 0:
        print("t = {t:.4f}".format(t = t))
    hignn_model.UpdateCoord(position)
    velocity = np.zeros(position.shape, dtype=np.float32)
    force = np.zeros((position.shape[0], 3), dtype=np.float32)
    force[:, 2] = -1.0
    
    hignn_model.Dot(velocity, force)
    
    return velocity

if __name__ == '__main__':    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    hignn.Init()
    
    # N = 216
    # nx = N
    # ny = N
    # nz = N
    # dx = 3
    # x = np.arange(0, nx * dx, dx)
    # y = np.arange(0, ny * dx, dx)
    # z = np.arange(0, nz * dx, dx)
    # xx, yy, zz = np.meshgrid(x, y, z)
    # X = np.concatenate(
    #     (xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)), axis=1)
    # X = X.astype(np.float32)
    
    X = np.loadtxt('output0.txt', dtype=np.float32)
    # X = np.loadtxt('output23.txt', dtype=np.float32)
    # X = np.loadtxt('Result/pos2.txt', dtype=np.float32)
    
    NN = X.shape[0]

    hignn_model = hignn.HignnModel(X, 100)
    
    hignn_model.LoadTwoBodyModel('nn/3D_force_UB_max600_try2')
    
    # set parameters for far dot, the following parameters are default values
    hignn_model.SetEpsilon(0.1)
    hignn_model.SetMaxIter(15)
    hignn_model.SetMatPoolSizeFactor(30)
    hignn_model.SetPostCheckFlag(False)
    hignn_model.SetUseSymmetryFlag(True)
    hignn_model.SetMaxFarDotWorkNodeSize(5000)
    hignn_model.SetMaxRelativeCoord(500000)
    
    # time_integrator = hignn.ExplicitEuler()
    
    # time_integrator.setTimeStep(0.0001)
    # time_integrator.setFinalTime(0.0001)
    # time_integrator.setNumRigidBody(NN)
    # time_integrator.setOutputStep(1)
    # time_integrator.initialize(X)
    # time_integrator.setVelocityFunc(velocity_update)
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
    for i in range(2):
        np.savetxt('Result/pos'+str(ite)+'rank'+str(rank)+'.txt', X[rank_range[rank]:rank_range[rank+1], :], fmt='%f', delimiter=' ')
        
        tt1 = time.time()
        V = velocity_update(ts, X)
        if rank == 0:
            print("Time for velocity_update: {t:.4f}s".format(t = time.time() - tt1))

        np.savetxt('Result/vel'+str(ite)+'rank'+str(rank)+'.txt', V[rank_range[rank]:rank_range[rank+1], :], fmt='%f', delimiter=' ')
        
        X = X + dt * V
        ts = ts + dt
        
        ite = ite + 1
        if rank == 0:
            print()            

    if rank == 0:
        print("Time for simulation: {t:.4f}s".format(t = time.time() - t1))
    
    # edgeInfo = hignn.BodyEdgeInfo()
    # edgeInfo.setThreeBodyEpsilon(5.0)

    # edgeInfo.setTargetSites(X)

    # edgeInfo.buildThreeBodyEdgeInfo()
    
    del hignn_model
    # del edgeInfo
    # del time_integrator
    
    hignn.Finalize()