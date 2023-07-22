#include "hignn.hpp"
#include "Typedef.hpp"

#include <chrono>
#include <fstream>

#include <mpi.h>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int mpiRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

  auto settings = Kokkos::InitializationSettings()
                      .set_num_threads(1)
                      .set_map_device_id_by("mpi_rank")
                      .set_disable_warnings(false);

  Kokkos::initialize(settings);

  std::chrono::time_point<std::chrono::system_clock> start, end;

  {
    const int nDim = 3;
    const int N = 100;
    const int nx = N;
    const int ny = N;
    const int nz = N;

    const double dx = 3.0;

    const int NN = nx * ny * nz;

    DeviceFloatMatrix coord;
    Kokkos::resize(coord, NN, 3);
    DeviceFloatMatrix::HostMirror coordMirror =
        Kokkos::create_mirror_view(coord);

    for (int i = 0; i < nx; i++)
      for (int j = 0; j < ny; j++)
        for (int k = 0; k < nz; k++) {
          int offset = i * ny * nz + j * nz + k;

          coordMirror(offset, 0) = i * dx + 0.2 * (float)rand() / RAND_MAX;
          coordMirror(offset, 1) = j * dx + 0.2 * (float)rand() / RAND_MAX;
          coordMirror(offset, 2) = k * dx + 0.2 * (float)rand() / RAND_MAX;
        }

    Kokkos::deep_copy(coord, coordMirror);
    // workflow: 1. build cluster tree for two coords, both rwo and col
    //           2. build the close, far relation
    //           3. do the dot operation

    // build the cluster trees
    Problem problem(coord, 100);

    MPI_Barrier(MPI_COMM_WORLD);
    start = std::chrono::system_clock::now();
    problem.Update();

    // Kokkos::deep_copy(coordMirror, coord);
    // std::ofstream vtkStream;
    // vtkStream.open("output.vtk", std::ios::out | std::ios::trunc);

    // vtkStream << "# vtk DataFile Version 2.0" << std::endl;

    // vtkStream << "output " << std::endl;

    // vtkStream << "ASCII" << std::endl << std::endl;

    // vtkStream << "DATASET POLYDATA" << std::endl
    //           << "POINTS " << NN << " float" << std::endl;

    // for (int i = 0; i < NN; i++) {
    //   for (int j = 0; j < 3; j++)
    //     vtkStream << coordMirror(i, j) << " ";
    //   vtkStream << std::endl;
    // }

    // vtkStream << "POINT_DATA " << NN << std::endl;

    // vtkStream << "SCALARS idx int 1" << std::endl
    //           << "LOOKUP_TABLE default" << std::endl;

    // for (int i = 0; i < NN; i++) {
    //   vtkStream << i << std::endl;
    // }

    // vtkStream.close();

    DeviceDoubleMatrix f, u;
    Kokkos::resize(f, NN, 3);
    Kokkos::resize(u, NN, 3);

    DeviceDoubleMatrix::HostMirror fMirror = Kokkos::create_mirror_view(f);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < NN; i++)
      for (int j = 0; j < 3; j++)
        fMirror(i, j) = 1.0;

    Kokkos::deep_copy(f, fMirror);

    problem.Dot(u, f);

    MPI_Barrier(MPI_COMM_WORLD);
    end = std::chrono::system_clock::now();
  }

  std::chrono::duration<double> elapsed_seconds = end - start;

  if (mpiRank == 0)
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

  Kokkos::finalize();

  MPI_Finalize();
}