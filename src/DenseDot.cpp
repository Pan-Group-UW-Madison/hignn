#include "HignnModel.hpp"

void HignnModel::DenseDot(DeviceDoubleMatrix u, DeviceDoubleMatrix f) {
  if (mMPIRank == 0)
    std::cout << "start of DenseDot" << std::endl;

  double queryDuration = 0;
  double dotDuration = 0;

  const std::size_t totalLeafNodeSize = mLeafNodeList.size();

  std::size_t leafNodeStart = 0, leafNodeEnd;
  for (unsigned int i = 0; i < (unsigned int)mMPIRank; i++) {
    std::size_t rankLeafNodeSize =
        totalLeafNodeSize / mMPISize + (i < totalLeafNodeSize % mMPISize);
    leafNodeStart += rankLeafNodeSize;
  }
  leafNodeEnd =
      leafNodeStart + totalLeafNodeSize / mMPISize +
      ((unsigned int)mMPIRank < totalLeafNodeSize % (unsigned int)mMPISize);
  leafNodeEnd = std::min(leafNodeEnd, totalLeafNodeSize);

  const std::size_t leafNodeSize = leafNodeEnd - leafNodeStart;

  const std::size_t maxWorkSize = std::min<std::size_t>(
      leafNodeSize, mMaxRelativeCoord / (mBlockSize * mBlockSize));
  int workSize = maxWorkSize;

  std::size_t totalNumQuery = 0;
  std::size_t totalNumIter = 0;

  DeviceFloatVector relativeCoordPool(
      "relativeCoordPool", maxWorkSize * mBlockSize * mBlockSize * 3);
  DeviceFloatMatrix queryResultPool("queryResultPool",
                                    maxWorkSize * mBlockSize * mBlockSize, 3);

  DeviceIntVector workingNode("workingNode", maxWorkSize);
  DeviceIntVector workingNodeOffset("workingNodeOffset", maxWorkSize);
  DeviceIntVector workingNodeCpy("workingNodeCpy", maxWorkSize);
  DeviceIntVector workingFlag("workingFlag", maxWorkSize);
  DeviceIntVector workingNodeCol("workingNodeCol", maxWorkSize);

  DeviceIntVector relativeCoordSize("relativeCoordSize", maxWorkSize);
  DeviceIntVector relativeCoordOffset("relativeCoordOffset", maxWorkSize);

  DeviceIndexVector nodeOffset("nodeOffset", leafNodeSize);

  auto &mCoord = *mCoordPtr;
  auto &mClusterTree = *mClusterTreePtr;

  DeviceIntVector mLeafNode("mLeafNode", totalLeafNodeSize);

  DeviceIntVector::HostMirror hostLeafNode =
      Kokkos::create_mirror_view(mLeafNode);

  for (size_t i = 0; i < totalLeafNodeSize; i++) {
    hostLeafNode(i) = mLeafNodeList[i];
  }
  Kokkos::deep_copy(mLeafNode, hostLeafNode);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, leafNodeSize),
      KOKKOS_LAMBDA(const std::size_t i) { nodeOffset(i) = 0; });

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workSize),
      KOKKOS_LAMBDA(const std::size_t i) {
        workingNode(i) = i;
        workingFlag(i) = 1;
      });

  int workingFlagSum = workSize;
  while (workSize > 0) {
    totalNumIter++;
    int totalCoord = 0;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workSize),
        KOKKOS_LAMBDA(const std::size_t i, int &tSum) {
          const int rank = i;
          const int node = workingNode(rank);
          const int nodeI = mLeafNode(node + leafNodeStart);
          const int nodeJ = mLeafNode(nodeOffset(node));

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
          const int nodeI = mLeafNode(node + leafNodeStart);
          const int nodeJ = mLeafNode(nodeOffset(node));
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
          const int nodeI = mLeafNode(node + leafNodeStart);
          const int nodeJ = mLeafNode(nodeOffset(node));
          const int relativeOffset = relativeCoordOffset(rank);

          const int indexIStart = mClusterTree(nodeI, 2);
          const int indexIEnd = mClusterTree(nodeI, 3);
          const int indexJStart = mClusterTree(nodeJ, 2);
          const int indexJEnd = mClusterTree(nodeJ, 3);

          const int workSizeI = indexIEnd - indexIStart;
          const int workSizeJ = indexJEnd - indexJStart;

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(teamMember, workSizeI * workSizeJ),
              [&](const std::size_t index) {
                const std::size_t j = index / workSizeJ;
                const std::size_t k = index % workSizeJ;
                for (int row = 0; row < 3; row++) {
                  double sum = 0.0;
                  for (int col = 0; col < 3; col++)
                    sum +=
                        0.5 *
                        (dataPtr[9 * (relativeOffset + index) + row * 3 + col] +
                         dataPtr[9 * (relativeOffset + index) + col * 3 +
                                 row]) *
                        f(indexJStart + k, col);
                  Kokkos::atomic_add(&u(indexIStart + j, row), sum);
                }
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
          if (nodeOffset(workingNode(rank)) == totalLeafNodeSize) {
            workingNode(rank) += maxWorkSize;
          }

          if (workingNode(rank) >= (int)leafNodeSize) {
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

  if (mMPIRank == 0) {
    std::cout << "query duration: " << queryDuration / 1e6 << "s" << std::endl;
    std::cout << "dot product duration: " << dotDuration / 1e6 << "s"
              << std::endl;
    std::cout << "totalNumQuery: " << totalNumQuery << std::endl;
    std::cout << "End of DenseDot." << std::endl;
  }
}