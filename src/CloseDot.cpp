#include <algorithm>

#include "HignnModel.hpp"

using namespace std;

void HignnModel::CloseDot(DeviceDoubleMatrix u, DeviceDoubleMatrix f) {
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  // Captures the current time to start measuring elapsed time for performance
  // tracking.

  if (mMPIRank == 0)
    std::cout << "start of CloseDot" << std::endl;

  // Timing variables to track the execution duration of query and dot
  // operations.
  double queryDuration = 0;
  double dotDuration = 0;

  // Set the total number of close node pairs and the maximum size of the batch
  // that will be processed at once.
  const int closeNodeSize = mCloseMatIPtr->extent(0);
  // Stores the number of close node pairs to be processed.
  const int maxWorkSize = 1000;  // Maximum close node pairs per batch.
  int workSize = std::min(maxWorkSize, closeNodeSize);
  // Close node pairs for current batch, constrained by
  // maxWorkSize and the number of close node pairs.
  int finishedNodeSize = 0;  // Number of node pairs that have been processed.

  // Variables to track the total number of queries and iterations processed.
  std::size_t totalNumQuery = 0;
  std::size_t totalNumIter = 0;

  // Vectors for storing relative coordinates and node work assignments.
  DeviceFloatVector relativeCoordPool("relativeCoordPool",
                                      mMaxRelativeCoord * 3);
  // Stores the relative coordinates of the particles.

  DeviceIntVector workingNode("workingNode", maxWorkSize);
  // Holds node indices for the current batch.

  DeviceIntVector relativeCoordSize("relativeCoordSize", maxWorkSize);
  // Stores work size for each node pair.
  DeviceIntVector relativeCoordOffset("relativeCoordOffset", maxWorkSize);

  auto &mCloseMatI = *mCloseMatIPtr;
  auto &mCloseMatJ = *mCloseMatJPtr;
  auto &mCoord = *mCoordPtr;
  auto &mClusterTree = *mClusterTreePtr;

  bool useSymmetry = mUseSymmetry;

  // Begin processing node pairs in batches
  while (finishedNodeSize < closeNodeSize) {
    {
      workSize = min(maxWorkSize, closeNodeSize - finishedNodeSize);
      // Update work size based on remaining node pairs.

      // Define bounds for adjusting work size.
      int lowerWorkSize = 0;
      int upperWorkSize = workSize;

      // Dynamically adjust work size based on estimated workload.
      while (true) {
        int estimatedWorkload = 0;
        // Variable to store the estimated workload for the current batch.

        // Parallel reduction to estimate workload by summing the work sizes of
        // node pairs.
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

              tSum += workSizeI * workSizeJ;  // Update the total estimated
                                              // workload for the batch.
            },
            Kokkos::Sum<int>(estimatedWorkload));

        // Adjustment of work size if estimated workload exceeds the maximum
        // allowed.
        if (estimatedWorkload > (int)mMaxRelativeCoord) {
          upperWorkSize = workSize;
          workSize = (lowerWorkSize + upperWorkSize) / 2;
          // Refine work size to distribute workload.
        } else {
          if (upperWorkSize - lowerWorkSize <= 1) {
            workSize = max(1, lowerWorkSize);
            // Finalize work size if difference between bounds is small.
            break;
          } else {
            lowerWorkSize = workSize;
            workSize = (lowerWorkSize + upperWorkSize) / 2;
            // Continue refining work size.
          }
        }
      }
    }

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workSize),
        KOKKOS_LAMBDA(const std::size_t i) {
          workingNode(i) = i + finishedNodeSize;
          // Assign node pairs to be processed in this batch.
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
          // Store the work size for this batch.

          tSum += workSizeI * workSizeJ;
          // Update the total coordinate count.
        },
        Kokkos::Sum<int>(totalCoord));
    Kokkos::fence();

    totalNumQuery += totalCoord;
    // Update the total number of queries.

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workSize),
        KOKKOS_LAMBDA(const int rank) {
          relativeCoordOffset(rank) = 0;
          for (int i = 0; i < rank; i++) {
            relativeCoordOffset(rank) += relativeCoordSize(i);
            // Calculate offset for relative coordinates.
          }
        });
    Kokkos::fence();

    // Calculate the relative coordinates.
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
                  // Calculate relative coordinate.
                }
              });
        });
    Kokkos::fence();

    // prepare the inference model.
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
                  // Accumulate results to u.
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
                      // Perform symmetry-based updates to u.
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
    // Update the count of processed node pairs.
  }

  MPI_Allreduce(MPI_IN_PLACE, &totalNumQuery, 1, MPI_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  // Aggregate total number of queries across all MPI processes.
  MPI_Allreduce(MPI_IN_PLACE, &totalNumIter, 1, MPI_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  ;  // Aggregate total number of iterations across all MPI processes.
  MPI_Allreduce(MPI_IN_PLACE, &queryDuration, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);
  // Aggregate query duration across all MPI processes.
  MPI_Allreduce(MPI_IN_PLACE, &dotDuration, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);
  // Aggregate dot duration across all MPI processes.

  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  // End the timer for performance tracking.

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