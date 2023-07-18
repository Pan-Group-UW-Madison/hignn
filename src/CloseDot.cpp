#include "hignn.hpp"

void Problem::CloseDot(DeviceFloatMatrix u, DeviceFloatMatrix f) {
  std::cout << "start of CloseDot" << std::endl;

  double queryDuration = 0;
  double dotDuration = 0;

  const int leafNodeSize = mLeafNodePtr->extent(0);
  const int maxWorkSize = std::min(100, leafNodeSize);
  int workSize = maxWorkSize;

  std::size_t totalNumQuery = 0;
  std::size_t totalNumIter = 0;

  DeviceFloatMatrix relativeCoordPool("relativeCoordPool",
                                      maxWorkSize * mBlockSize * mBlockSize, 3);
  DeviceFloatMatrix queryResultPool("queryResultPool",
                                    maxWorkSize * mBlockSize * mBlockSize, 3);

  DeviceIntVector workingNode("workingNode", maxWorkSize);
  DeviceIntVector workingNodeOffset("workingNodeOffset", maxWorkSize);
  DeviceIntVector workingNodeCpy("workingNodeCpy", maxWorkSize);
  DeviceIntVector workingFlag("workingFlag", maxWorkSize);

  DeviceIntVector relativeCoordSize("relativeCoordSize", maxWorkSize);
  DeviceIntVector relativeCoordOffset("relativeCoordOffset", maxWorkSize);

  DeviceIndexVector nodeOffset("nodeOffset", leafNodeSize);

  auto &mCloseMatI = *mCloseMatIPtr;
  auto &mCloseMatJ = *mCloseMatJPtr;
  auto &mCoord = *mCoordPtr;
  auto &mClusterTree = *mClusterTreePtr;
  auto &mLeafNode = *mLeafNodePtr;

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, leafNodeSize),
      KOKKOS_LAMBDA(const std::size_t i) { nodeOffset(i) = mCloseMatI(i); });

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workSize),
      KOKKOS_LAMBDA(const std::size_t i) {
        workingNode(i) = i;
        workingFlag(i) = 1;
      });

  const int blockSize = mBlockSize;
  int workingFlagSum = workSize;
  while (workSize > 0) {
    totalNumIter++;
    int totalCoord = 0;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workSize),
        KOKKOS_LAMBDA(const std::size_t i, int &tSum) {
          const int rank = i;
          const int node = workingNode(rank);
          const int nodeI = mLeafNode(node);
          const int nodeJ = mCloseMatJ(nodeOffset(node));
          const int relativeOffset = rank * blockSize * blockSize;

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
          const int nodeI = mLeafNode(node);
          const int nodeJ = mCloseMatJ(nodeOffset(node));
          const int relativeOffset = relativeCoordOffset(rank);

          const int indexIStart = mClusterTree(nodeI, 2);
          const int indexIEnd = mClusterTree(nodeI, 3);
          const int indexJStart = mClusterTree(nodeJ, 2);
          const int indexJEnd = mClusterTree(nodeJ, 3);

          const int workSizeI = indexIEnd - indexIStart;
          const int workSizeJ = indexJEnd - indexJStart;

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(teamMember, workSizeI), [&](const int j) {
                Kokkos::parallel_for(
                    Kokkos::ThreadVectorRange(teamMember, workSizeJ),
                    [&](const int k) {
                      const int index = relativeOffset + j * workSizeJ + k;
                      for (int l = 0; l < 3; l++) {
                        relativeCoordPool(index, l) =
                            mCoord(indexJStart + k, l) -
                            mCoord(indexIStart + j, l);
                      }
                    });
              });
        });
    Kokkos::fence();

    // do inference
    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(torch::kCUDA, 0)
                       .requires_grad(false);
    torch::Tensor relativeCoordTensor =
        torch::from_blob(relativeCoordPool.data(), {totalCoord, 3}, options);
    std::vector<c10::IValue> inputs;
    inputs.push_back(relativeCoordTensor);

    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();

    auto resultTensor = module.forward(inputs).toTensor();

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
          const int nodeI = mLeafNode(node);
          const int nodeJ = mCloseMatJ(nodeOffset(node));
          const int relativeOffset = relativeCoordOffset(rank);

          const int indexIStart = mClusterTree(nodeI, 2);
          const int indexIEnd = mClusterTree(nodeI, 3);
          const int indexJStart = mClusterTree(nodeJ, 2);
          const int indexJEnd = mClusterTree(nodeJ, 3);

          const int workSizeI = indexIEnd - indexIStart;
          const int workSizeJ = indexJEnd - indexJStart;

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(teamMember, workSizeI), [&](const int j) {
                Kokkos::parallel_for(
                    Kokkos::ThreadVectorRange(teamMember, workSizeJ),
                    [&](const int k) {
                      int index = relativeOffset + j * workSizeJ + k;
                      // there is an exception here
                      // calculate r
                      float r = 0;
                      for (int l = 0; l < 3; l++) {
                        r += relativeCoordPool(index, l) *
                             relativeCoordPool(index, l);
                      }
                      r = sqrt(r);
                      if (r < 1e-6) {
                        dataPtr[9 * index] = 1.0;
                        dataPtr[9 * index + 1] = 0.0;
                        dataPtr[9 * index + 2] = 0.0;
                        dataPtr[9 * index + 3] = 0.0;
                        dataPtr[9 * index + 4] = 1.0;
                        dataPtr[9 * index + 5] = 0.0;
                        dataPtr[9 * index + 6] = 0.0;
                        dataPtr[9 * index + 7] = 0.0;
                        dataPtr[9 * index + 8] = 1.0;
                      }
                      for (int row = 0; row < 3; row++)
                        for (int col = 0; col < 3; col++)
                          u(indexIStart + j, row) +=
                              dataPtr[9 * index + row * 3 + col] *
                              f(indexJStart + j, col);
                    });
              });
        });
    Kokkos::fence();

    end = std::chrono::steady_clock::now();
    dotDuration +=
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
            .count();

    // post processing
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workSize),
        KOKKOS_LAMBDA(const int rank) { workingFlag(rank) = 1; });
    Kokkos::fence();

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workSize),
        KOKKOS_LAMBDA(const int rank) {
          nodeOffset(workingNode(rank))++;
          if (nodeOffset(workingNode(rank)) ==
              mCloseMatI(workingNode(rank) + 1)) {
            workingNode(rank) += maxWorkSize;
          }

          if (workingNode(rank) >= leafNodeSize) {
            workingFlag(rank) = 0;
          }
        });
    Kokkos::fence();

    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workSize),
        KOKKOS_LAMBDA(const std::size_t i, int &tSum) {
          tSum += workingFlag(i);
        },
        Kokkos::Sum<int>(workingFlagSum));
    Kokkos::fence();

    if (workSize > workingFlagSum) {
      // copy the working node to working node cpy and shrink the work size
      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workSize),
          KOKKOS_LAMBDA(const int rank) {
            workingNodeCpy(rank) = workingNode(rank);
            int counter = rank + 1;
            if (rank < workingFlagSum)
              for (int i = 0; i < workSize; i++) {
                if (workingFlag(i) == 1) {
                  counter--;
                }
                if (counter == 0) {
                  workingNodeOffset(rank) = i;
                  break;
                }
              }
          });
      Kokkos::fence();

      workSize = workingFlagSum;

      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workSize),
          KOKKOS_LAMBDA(const int rank) {
            workingNode(rank) = workingNodeCpy(workingNodeOffset(rank));
          });
      Kokkos::fence();
    }
  }

  printf("num query: %ld, num iteration: %ld, query duration: %.4fs, dot "
         "duration: %.4fs\n",
         totalNumQuery, totalNumIter, queryDuration / 1e6, dotDuration / 1e6);
  std::cout << "end of CloseDot" << std::endl;
}