#include <algorithm>

#include "HignnModel.hpp"

using namespace std;

void HignnModel::CloseDot(DeviceDoubleMatrix u, DeviceDoubleMatrix f) {
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

  if (mMPIRank == 0)
    std::cout << "start of CloseDot" << std::endl;

  double queryDuration = 0;
  double dotDuration = 0;

  const int closeNodeSize = mCloseMatIPtr->extent(0);
  const int maxWorkSize = 1000;
  int workSize = std::min(maxWorkSize, closeNodeSize);
  int finishedNodeSize = 0;

  std::size_t totalNumQuery = 0;
  std::size_t totalNumIter = 0;

  DeviceFloatVector relativeCoordPool("relativeCoordPool",
                                      mMaxRelativeCoord * 3);

  DeviceIntVector workingNode("workingNode", maxWorkSize);

  DeviceIntVector relativeCoordSize("relativeCoordSize", maxWorkSize);
  DeviceIntVector relativeCoordOffset("relativeCoordOffset", maxWorkSize);

  auto &mCloseMatI = *mCloseMatIPtr;
  auto &mCloseMatJ = *mCloseMatJPtr;
  auto &mCoord = *mCoordPtr;
  auto &mClusterTree = *mClusterTreePtr;

  bool useSymmetry = mUseSymmetry;

  while (finishedNodeSize < closeNodeSize) {
    {
      workSize = min(maxWorkSize, closeNodeSize - finishedNodeSize);

      int lowerWorkSize = 0;
      int upperWorkSize = workSize;

      while (true) {
        int estimatedWorkload = 0;

        Kokkos::parallel_reduce(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workSize),
            KOKKOS_LAMBDA(const std::size_t i, int &tSum) {
              const int nodeI = mCloseMatI(i + finishedNodeSize);
              const int indexIStart = mClusterTree(nodeI, 2);
              const int indexIEnd = mClusterTree(nodeI, 3);
              const int workSizeI = indexIEnd - indexIStart;

              const int nodeJ = mCloseMatJ(i + finishedNodeSize);
              const int indexJStart = mClusterTree(nodeJ, 2);
              const int indexJEnd = mClusterTree(nodeJ, 3);
              const int workSizeJ = indexJEnd - indexJStart;

              tSum += workSizeI * workSizeJ;
            },
            Kokkos::Sum<int>(estimatedWorkload));

        if (estimatedWorkload > (int)mMaxRelativeCoord) {
          upperWorkSize = workSize;
          workSize = (lowerWorkSize + upperWorkSize) / 2;
        } else {
          if (upperWorkSize - lowerWorkSize <= 1) {
            workSize = lowerWorkSize;
            break;
          } else {
            lowerWorkSize = workSize;
            workSize = (lowerWorkSize + upperWorkSize) / 2;
          }
        }
      }
    }

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workSize),
        KOKKOS_LAMBDA(const std::size_t i) {
          workingNode(i) = i + finishedNodeSize;
        });
    Kokkos::fence();

    totalNumIter++;
    int totalCoord = 0;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workSize),
        KOKKOS_LAMBDA(const std::size_t i, int &tSum) {
          const int rank = i;
          const int node = workingNode(rank);
          const int nodeI = mCloseMatI(node);
          const int nodeJ = mCloseMatJ(node);

          const int indexIStart = mClusterTree(nodeI, 2);
          const int indexIEnd = mClusterTree(nodeI, 3);
          const int indexJStart = mClusterTree(nodeJ, 2);
          const int indexJEnd = mClusterTree(nodeJ, 3);

          const int workSizeI = indexIEnd - indexIStart;
          const int workSizeJ = indexJEnd - indexJStart;

          relativeCoordSize(rank) = workSizeI * workSizeJ;

          tSum += workSizeI * workSizeJ;
        },
        Kokkos::Sum<int>(totalCoord));
    Kokkos::fence();

    totalNumQuery += totalCoord;
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workSize),
        KOKKOS_LAMBDA(const int rank) {
          relativeCoordOffset(rank) = 0;
          for (int i = 0; i < rank; i++) {
            relativeCoordOffset(rank) += relativeCoordSize(i);
          }
        });
    Kokkos::fence();

    // calculate the relative coordinates
    Kokkos::parallel_for(
        Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(workSize,
                                                          Kokkos::AUTO()),
        KOKKOS_LAMBDA(
            const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type
                &teamMember) {
          const int rank = teamMember.league_rank();
          const int node = workingNode(rank);
          const int nodeI = mCloseMatI(node);
          const int nodeJ = mCloseMatJ(node);
          const int relativeOffset = relativeCoordOffset(rank);

          const int indexIStart = mClusterTree(nodeI, 2);
          const int indexIEnd = mClusterTree(nodeI, 3);
          const int indexJStart = mClusterTree(nodeJ, 2);
          const int indexJEnd = mClusterTree(nodeJ, 3);

          const int workSizeI = indexIEnd - indexIStart;
          const int workSizeJ = indexJEnd - indexJStart;

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(teamMember, workSizeI * workSizeJ),
              [&](const int i) {
                int j = i / workSizeJ;
                int k = i % workSizeJ;

                const int index = relativeOffset + j * workSizeJ + k;
                for (int l = 0; l < 3; l++) {
                  relativeCoordPool(3 * index + l) =
                      mCoord(indexJStart + k, l) - mCoord(indexIStart + j, l);
                }
              });
        });
    Kokkos::fence();

    // do inference
#if USE_GPU
    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(torch::kCUDA, mCudaDevice)
                       .requires_grad(false);
#else
    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(torch::kCPU)
                       .requires_grad(false);
#endif
    torch::Tensor relativeCoordTensor =
        torch::from_blob(relativeCoordPool.data(), {totalCoord, 3}, options);
    std::vector<c10::IValue> inputs;
    inputs.push_back(relativeCoordTensor);

    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();

    auto resultTensor = mTwoBodyModel.forward(inputs).toTensor();

    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();
    queryDuration +=
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
            .count();

    begin = std::chrono::steady_clock::now();

    auto dataPtr = resultTensor.data_ptr<float>();

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(workSize,
                                                          Kokkos::AUTO()),
        KOKKOS_LAMBDA(
            const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type
                &teamMember) {
          const int rank = teamMember.league_rank();
          const int node = workingNode(rank);
          const int nodeI = mCloseMatI(node);
          const int nodeJ = mCloseMatJ(node);
          const int relativeOffset = relativeCoordOffset(rank);

          const std::size_t indexIStart = mClusterTree(nodeI, 2);
          const std::size_t indexIEnd = mClusterTree(nodeI, 3);
          const std::size_t indexJStart = mClusterTree(nodeJ, 2);
          const std::size_t indexJEnd = mClusterTree(nodeJ, 3);

          const std::size_t workSizeI = indexIEnd - indexIStart;
          const std::size_t workSizeJ = indexJEnd - indexJStart;

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(teamMember, workSizeI * workSizeJ),
              [&](const std::size_t index) {
                const std::size_t j = index / workSizeJ;
                const std::size_t k = index % workSizeJ;
                for (int row = 0; row < 3; row++) {
                  double sum = 0.0;
                  for (int col = 0; col < 3; col++)
                    sum +=
                        dataPtr[9 * (relativeOffset + index) + row * 3 + col] *
                        f(indexJStart + k, col);
                  Kokkos::atomic_add(&u(indexIStart + j, row), sum);
                }
              });

          if (useSymmetry)
            if (nodeJ > nodeI) {
              Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(teamMember, workSizeI * workSizeJ),
                  [&](const std::size_t index) {
                    const std::size_t j = index / workSizeJ;
                    const std::size_t k = index % workSizeJ;
                    for (int row = 0; row < 3; row++) {
                      double sum = 0.0;
                      for (int col = 0; col < 3; col++)
                        sum += dataPtr[9 * (relativeOffset + index) + row * 3 +
                                       col] *
                               f(indexIStart + j, col);
                      Kokkos::atomic_add(&u(indexJStart + k, row), sum);
                    }
                  });
            }
        });
    Kokkos::fence();

    end = std::chrono::steady_clock::now();
    dotDuration +=
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
            .count();

    finishedNodeSize += workSize;
  }

  MPI_Allreduce(MPI_IN_PLACE, &totalNumQuery, 1, MPI_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &totalNumIter, 1, MPI_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &queryDuration, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &dotDuration, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);

  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

  if (mMPIRank == 0) {
    printf(
        "num query: %ld, num iteration: %ld, query duration: %.4fs, dot "
        "duration: %.4fs\n",
        totalNumQuery, totalNumIter, queryDuration / 1e6, dotDuration / 1e6);
    printf(
        "End of close dot. Dot time %.4fs\n",
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() /
            1e6);
  }
}