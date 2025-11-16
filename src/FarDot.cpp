#include "HignnModel.hpp"

// Kokkos reduction for array data
struct ArrReduce {
  double values[6];

  // Initializes all values to zero
  KOKKOS_INLINE_FUNCTION ArrReduce() {
    for (int i = 0; i < 6; i++) {
      values[i] = 0;
    }
  }

  // Copy constructor
  KOKKOS_INLINE_FUNCTION ArrReduce(const ArrReduce &rhs) {
    for (int i = 0; i < 6; i++) {
      values[i] = rhs.values[i];
    }
  }

  // Addition operator for Kokkos reductions
  KOKKOS_INLINE_FUNCTION ArrReduce &operator+=(const ArrReduce &src) {
    for (int i = 0; i < 6; i++) {
      values[i] += src.values[i];
    }

    return *this;
  }
};

namespace Kokkos {
template <>
struct reduction_identity<ArrReduce> {
  KOKKOS_FORCEINLINE_FUNCTION static ArrReduce sum() {
    return ArrReduce();
  }
};
}  // namespace Kokkos

void HignnModel::FarDot(DeviceDoubleMatrix u, DeviceDoubleMatrix f) {
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  // Captures the current time to start measuring elapsed time for performance
  // tracking.

  if (mMPIRank == 0)
    std::cout << "start of FarDot" << std::endl;
  MPI_Barrier(MPI_COMM_WORLD);

  // Timing variables for profiling different parts of the algorithm.
  double queryDuration = 0.0;
  double dotDuration = 0.0;
  double ckNormalizationDuration = 0.0;
  double qkNormalizationDuration = 0.0;
  double stopCriterionDuration = 0.0;
  double resetDuration = 0.0;

  // Variables to track the number of queries, total iterations, and max inner
  // iterations.
  std::size_t totalNumQuery = 0;
  std::size_t totalNumIter = 0;
  int innerNumIter = 0;
  int maxInnerNumIter = 0;

  // Constants and working array sizes for batch and matrix pool management.
  const int maxRelativeCoord = mMaxRelativeCoord;
  const int matPoolSize = maxRelativeCoord * mMatPoolSizeFactor;
  const int maxWorkNodeSize = mMaxFarDotWorkNodeSize;
  const int maxIter = mMaxIter;
  const int middleMatPoolSize = maxWorkNodeSize * maxIter;
  int workNodeSize = 0;     // Number of node pairs in current batch.
  int allowedWorkload = 0;  // Maximum workload allowed in current batch.

  const bool postCheck = false;

  // Device arrays and matrix pools for intermediate computation and batching.
  DeviceFloatVector relativeCoordPool(
      "relativeCoordPool",
      maxRelativeCoord *
          3);  // Stores the relative coordinates of the particles.
  DeviceDoubleMatrix cMatPool("cMatPool", matPoolSize,
                              9);  // Pool for C matrices.
  DeviceDoubleMatrix qMatPool("qMatPool", matPoolSize,
                              9);  // Pool for Q matrices.
  DeviceDoubleVector middleMatPool(
      "middleMatPool",
      middleMatPoolSize * 3);  // Middle matrix pool for accumulation.
  DeviceDoubleMatrix ckMatPool("ckMatPool", maxRelativeCoord,
                               9);  // Temporary C matrices.
  DeviceDoubleMatrix ckInvMatPool("ckInvMatPool", maxWorkNodeSize,
                                  9);  // Inverses of temporary C matrices.

  DeviceIntVector workingNode(
      "workingNode",
      maxWorkNodeSize);  // Holds node indices for the current batch.
  DeviceIntVector dotProductNode(
      "dotProductNode",
      maxWorkNodeSize);  // Nodes for dot product in final step.
  DeviceIntVector dotProductRank(
      "dotProductRank", maxWorkNodeSize);  // Ranks for dot product nodes.
  DeviceIntVector stopNode(
      "stopNode", maxWorkNodeSize);  // Flags to signal stopping criterion.
  DeviceIntMatrix workingNodeCMatOffset(
      "workingNodeCMatOffset", maxWorkNodeSize,
      maxIter);  // Offset in cMatPool for each working node and each
                 // iteration.
  DeviceIntMatrix workingNodeQMatOffset(
      "workingNodeQMatOffset", maxWorkNodeSize,
      maxIter);  // Offset in qMatPool for each working node and each
                 // iteration.
  DeviceIntMatrix workingNodeSelectedColIdx("workingNodeSelectedColIdx",
                                            maxWorkNodeSize,
                                            maxIter);  // Column indices used.
  DeviceIntMatrix workingNodeSelectedRowIdx("workingNodeSelectedRowIdx",
                                            maxWorkNodeSize,
                                            maxIter);  // Row indices used.
  DeviceIntVector workingNodeIteration(
      "workingNodeIteration",
      maxWorkNodeSize);  // Iteration counters for nodes.

  DeviceIntVector workingNodeCopy(
      "workingNodeCopy",
      maxWorkNodeSize * maxIter);  // Used for copying node arrays.
  DeviceIntVector workingNodeCopyOffset(
      "workingNodeCopyOffset",
      maxWorkNodeSize);  // Copy offsets for node arrays.

  DeviceDoubleVector nu2("nu2",
                         maxWorkNodeSize);  // Stores power-iteration numerators
                                            // for stopping criterion.
  DeviceDoubleVector mu2(
      "mu", maxWorkNodeSize);  // Stores power-iteration denominators for
                               // stopping criterion.

  DeviceDoubleVector workingNodeDoubleCopy(
      "workingNodeDoubleCopy",
      maxWorkNodeSize);  // Used for copying mu2.

  DeviceIntVector uDotCheck(
      "uDotCheck", maxWorkNodeSize);  // Used for velocity post-checking.

  const double epsilon = mEpsilon;
  const double epsilon2 = epsilon * epsilon;

  DeviceIntVector relativeCoordOffset(
      "relativeCoordOffset",
      maxWorkNodeSize);  // Offsets for relative coordinates.

  // Short references for large data structures
  auto &mFarMatI = *mFarMatIPtr;
  auto &mFarMatJ = *mFarMatJPtr;
  auto &mCoord = *mCoordPtr;
  auto &mClusterTree = *mClusterTreePtr;

  const int farNodeSize = mFarMatI.extent(0);  // Number of far node pairs
  int finishedNodeSize = 0;  // Number of node pairs processed so far.
  int installedNode = 0;     // Next node index to be installed for batch.
  int totalCoord = 0;        // Tracks total coordinates processed in a batch.

  bool farDotFinished = false;

  /** Begin processing node pairs in adaptive batches */
  while (!farDotFinished) {
    totalNumIter++;
    innerNumIter++;

    // select working node
    if (workNodeSize == 0) {
      allowedWorkload = maxRelativeCoord;

      // estimate the workload and determine optimal batch size
      int estimatedWorkload;
      int leftNode = std::min(farNodeSize - finishedNodeSize, maxWorkNodeSize);
      workNodeSize = leftNode;
      int lowerWorkNodeSize = 0;
      int upperWorkNodeSize = workNodeSize;
      // install new working node
      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workNodeSize),
          KOKKOS_LAMBDA(const int i) { workingNode(i) = installedNode + i; });

      // Dynamically adjust batch size to avoid exceeding allowedWorkload
      while (true) {
        int estimatedQMatWorkload = 0;
        int estimatedCMatWorkload = 0;

        // Estimate total row workload for C matrices in the batch
        Kokkos::parallel_reduce(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workNodeSize),
            KOKKOS_LAMBDA(const std::size_t i, int &tSum) {
              int nodeI = mFarMatI(workingNode(i));
              int workload = mClusterTree(nodeI, 3) - mClusterTree(nodeI, 2);
              tSum += workload;
            },
            Kokkos::Sum<int>(estimatedCMatWorkload));

        // Estimate total column workload for Q matrices in the batch
        Kokkos::parallel_reduce(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workNodeSize),
            KOKKOS_LAMBDA(const std::size_t i, int &tSum) {
              int nodeJ = mFarMatJ(workingNode(i));
              int workload = mClusterTree(nodeJ, 3) - mClusterTree(nodeJ, 2);
              tSum += workload;
            },
            Kokkos::Sum<int>(estimatedQMatWorkload));

        // Use the larger workload size to check if the batch needs to be
        // split
        estimatedWorkload =
            std::max(estimatedCMatWorkload, estimatedQMatWorkload);

        if (estimatedWorkload > allowedWorkload) {
          upperWorkNodeSize = workNodeSize;
          workNodeSize = (upperWorkNodeSize + lowerWorkNodeSize) / 2;
        } else {
          if (upperWorkNodeSize - lowerWorkNodeSize <= 1) {
            break;
          } else {
            lowerWorkNodeSize = workNodeSize;
            workNodeSize = (upperWorkNodeSize + lowerWorkNodeSize) / 2;
          }
        }
      }

      installedNode += workNodeSize;

      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workNodeSize),
          KOKKOS_LAMBDA(const std::size_t i) {
            const int nodeJ = mFarMatJ(workingNode(i));
            int workload = mClusterTree(nodeJ, 3) - mClusterTree(nodeJ, 2);

            workingNodeIteration(i) = 0;
            workingNodeSelectedColIdx(i, 0) = workload / 2;

            nu2(i) = 0.0;
            mu2(i) = 0.0;
          });

      // estimate CMat offset
      totalCoord = 0;
      Kokkos::parallel_scan(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workNodeSize),
          KOKKOS_LAMBDA(const int i, int &update, const bool final) {
            if (final)
              relativeCoordOffset(i) = update;
            const int nodeI = mFarMatI(workingNode(i));

            const int indexIStart = mClusterTree(nodeI, 2);
            const int indexIEnd = mClusterTree(nodeI, 3);

            const int workSizeI = indexIEnd - indexIStart;

            update += workSizeI;
          },
          totalCoord);
      Kokkos::fence();

      // Fill workingNodeCMatOffset for C matrix placement
      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
              0, workNodeSize * maxIter),
          KOKKOS_LAMBDA(const int i) {
            const int rank = i / maxIter;
            const int l = i % maxIter;

            const int nodeI = mFarMatI(workingNode(rank));

            const int indexIStart = mClusterTree(nodeI, 2);
            const int indexIEnd = mClusterTree(nodeI, 3);

            const int workSizeI = indexIEnd - indexIStart;

            workingNodeCMatOffset(rank, l) =
                relativeCoordOffset(rank) * maxIter + workSizeI * l;
          });
      Kokkos::fence();

      // estimate QMat offset
      totalCoord = 0;
      Kokkos::parallel_scan(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workNodeSize),
          KOKKOS_LAMBDA(const int i, int &update, const bool final) {
            if (final)
              relativeCoordOffset(i) = update;
            const int nodeJ = mFarMatJ(workingNode(i));

            const int indexJStart = mClusterTree(nodeJ, 2);
            const int indexJEnd = mClusterTree(nodeJ, 3);

            const int workSizeJ = indexJEnd - indexJStart;

            update += workSizeJ;
          },
          totalCoord);
      Kokkos::fence();

      // Fill workingNodeCMatOffset for C matrix placement
      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
              0, workNodeSize * maxIter),
          KOKKOS_LAMBDA(const int i) {
            const int rank = i / maxIter;
            const int l = i % maxIter;

            const int nodeJ = mFarMatJ(workingNode(rank));

            const int indexJStart = mClusterTree(nodeJ, 2);
            const int indexJEnd = mClusterTree(nodeJ, 3);

            const int workSizeJ = indexJEnd - indexJStart;

            workingNodeQMatOffset(rank, l) =
                relativeCoordOffset(rank) * maxIter + workSizeJ * l;
          });
      Kokkos::fence();
    }

    {
      std::chrono::steady_clock::time_point begin =
          std::chrono::steady_clock::now();

      // calculate relative coord for C
      totalCoord = 0;
      Kokkos::parallel_scan(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workNodeSize),
          KOKKOS_LAMBDA(const int i, int &update, const bool final) {
            if (final)
              relativeCoordOffset(i) = update;
            const int nodeI = mFarMatI(workingNode(i));

            const int indexIStart = mClusterTree(nodeI, 2);
            const int indexIEnd = mClusterTree(nodeI, 3);

            const int workSizeI = indexIEnd - indexIStart;

            update += workSizeI;
          },
          totalCoord);
      totalNumQuery += totalCoord;

      // calculate the relative coordinates
      Kokkos::parallel_for(
          Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(workNodeSize,
                                                            Kokkos::AUTO()),
          KOKKOS_LAMBDA(
              const Kokkos::TeamPolicy<
                  Kokkos::DefaultExecutionSpace>::member_type &teamMember) {
            const int rank = teamMember.league_rank();
            const int nodeI = mFarMatI(workingNode(rank));
            const int relativeOffset = relativeCoordOffset(rank);

            const int indexIStart = mClusterTree(nodeI, 2);
            const int indexIEnd = mClusterTree(nodeI, 3);
            const int workSizeI = indexIEnd - indexIStart;

            const int nodeJ = mFarMatJ(workingNode(rank));
            const int indexJ =
                mClusterTree(nodeJ, 2) +
                workingNodeSelectedColIdx(rank, workingNodeIteration(rank));

            Kokkos::parallel_for(Kokkos::TeamVectorRange(teamMember, workSizeI),
                                 [&](const int j) {
                                   const int index = relativeOffset + j;
                                   for (int l = 0; l < 3; l++) {
                                     relativeCoordPool(3 * index + l) =
                                         mCoord(indexJ, l) -
                                         mCoord(indexIStart + j, l);
                                   }
                                 });
          });
      Kokkos::fence();

      // do inference for CMat
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

      auto resultTensor = mTwoBodyModel.forward(inputs).toTensor();

      // copy result to CMat
      auto dataPtr = resultTensor.data_ptr<float>();

      // Fill cMatPool with model outputs, enforcing symmetry on non-diagonal
      // elements.
      Kokkos::parallel_for(
          Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(workNodeSize,
                                                            Kokkos::AUTO()),
          KOKKOS_LAMBDA(
              const Kokkos::TeamPolicy<
                  Kokkos::DefaultExecutionSpace>::member_type &teamMember) {
            const int rank = teamMember.league_rank();
            const int nodeI = mFarMatI(workingNode(rank));
            const int relativeOffset = relativeCoordOffset(rank);

            const int indexIStart = mClusterTree(nodeI, 2);
            const int indexIEnd = mClusterTree(nodeI, 3);
            const int workSizeI = indexIEnd - indexIStart;

            const int ckOffset =
                workingNodeCMatOffset(rank, workingNodeIteration(rank));

            // Loop over each row in the current batch to copy the predicted C
            // matrices to cMatPool. Enforce symmetry by averaging each
            // off-diagonal element with its transpose.
            Kokkos::parallel_for(Kokkos::TeamVectorRange(teamMember, workSizeI),
                                 [&](const int j) {
                                   const int index = relativeOffset + j;

                                   for (int row = 0; row < 3; row++)
                                     for (int col = 0; col < 3; col++)
                                       if (row == col)
                                         cMatPool(ckOffset + j, 3 * row + col) =
                                             dataPtr[index * 9 + 3 * row + col];
                                       else {
                                         const int l1 = 3 * row + col;
                                         const int l2 = 3 * col + row;
                                         cMatPool(ckOffset + j, l1) =
                                             0.5 * (dataPtr[index * 9 + l1] +
                                                    dataPtr[index * 9 + l2]);
                                       }
                                 });
          });
      Kokkos::fence();

      std::chrono::steady_clock::time_point end =
          std::chrono::steady_clock::now();
      queryDuration +=
          std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
              .count();
    }

    // find index for QMat
    {
      auto begin = std::chrono::steady_clock::now();

      // Step 1: Orthogonalize the current C matrix (remove projections on
      // previous bases)
      Kokkos::parallel_for(
          Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(workNodeSize,
                                                            Kokkos::AUTO()),
          KOKKOS_LAMBDA(
              const Kokkos::TeamPolicy<
                  Kokkos::DefaultExecutionSpace>::member_type &teamMember) {
            const int rank = teamMember.league_rank();
            const int nodeI = mFarMatI(workingNode(rank));
            const int indexIStart = mClusterTree(nodeI, 2);
            const int indexIEnd = mClusterTree(nodeI, 3);

            const int rowSize = indexIEnd - indexIStart;

            // Offset for current C matrix and selected column index for Q
            const int ckOffset =
                workingNodeCMatOffset(rank, workingNodeIteration(rank));
            const int jk =
                workingNodeSelectedColIdx(rank, workingNodeIteration(rank));

            // Loop over all entries of the C matrix for the current node
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(teamMember, rowSize * 9),
                [&](const int j) {
                  const int index = ckOffset + j / 9;
                  const int k = j % 9;
                  const int row = k / 3;
                  const int col = k % 3;

                  double sum = 0.0;
                  for (int l = 0; l < innerNumIter - 1; l++) {
                    const int indexL = workingNodeCMatOffset(rank, l) + j / 9;
                    const int qMatOffsetJk =
                        workingNodeQMatOffset(rank, l) + jk;
                    for (int m = 0; m < 3; m++)
                      sum += cMatPool(indexL, 3 * row + m) *
                             qMatPool(qMatOffsetJk, 3 * m + col);
                  }
                  cMatPool(index, k) -= sum;
                });
          });
      Kokkos::fence();

      // Step 2: Find the row of the current C matrix block with the largest
      // determinant (determinant is better than norm according to the in-house
      // test), and store the corresponding index. Also compute and store the
      // inverse.
      Kokkos::parallel_for(
          Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(workNodeSize,
                                                            Kokkos::AUTO()),
          KOKKOS_LAMBDA(
              const Kokkos::TeamPolicy<
                  Kokkos::DefaultExecutionSpace>::member_type &teamMember) {
            const int rank = teamMember.league_rank();

            const int nodeI = mFarMatI(workingNode(rank));

            const int rowSize = mClusterTree(nodeI, 3) - mClusterTree(nodeI, 2);

            const int ckOffset =
                workingNodeCMatOffset(rank, workingNodeIteration(rank));

            // Result will hold the max determinant value and its corresponding
            // index
            Kokkos::MaxLoc<double, int>::value_type result;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(teamMember, rowSize),
                [&](const int j,
                    Kokkos::MaxLoc<double, int>::value_type &update) {
                  double curNorm = 0.0;
                  const int index = ckOffset + j;
                  const int ckIndex = relativeCoordOffset(rank) + j;

                  // Compute determinant for current 3x3 matrix block
                  const double a = cMatPool(index, 0);
                  const double b = cMatPool(index, 1);
                  const double c = cMatPool(index, 2);
                  const double d = cMatPool(index, 3);
                  const double e = cMatPool(index, 4);
                  const double f = cMatPool(index, 5);
                  const double g = cMatPool(index, 6);
                  const double h = cMatPool(index, 7);
                  const double i = cMatPool(index, 8);

                  curNorm = a * (e * i - f * h) - b * (d * i - f * g) +
                            c * (d * h - e * g);

                  // Store a copy of the current C matrix block for later use
                  for (int k = 0; k < 9; k++)
                    ckMatPool(ckIndex, k) = cMatPool(index, k);

                  bool exist = false;
                  for (int l = 0; l < workingNodeIteration(rank); l++)
                    if (workingNodeSelectedRowIdx(rank, l) == j)
                      exist = true;

                  // Update result if current determinant is larger and row is
                  // unused
                  if (curNorm > update.val && exist == false) {
                    update.val = curNorm;
                    update.loc = j;
                  }
                },
                Kokkos::MaxLoc<double, int>(result));
            teamMember.team_barrier();

            // Store the index of the row with the largest determinant for this
            // node/iteration
            workingNodeSelectedRowIdx(rank, workingNodeIteration(rank)) =
                result.loc;

            const int ik = result.loc;

            // Compute and store the inverse of the selected C matrix block
            Kokkos::single(Kokkos::PerTeam(teamMember), [&]() {
              const double a = cMatPool(ckOffset + ik, 0);
              const double b = cMatPool(ckOffset + ik, 1);
              const double c = cMatPool(ckOffset + ik, 2);
              const double d = cMatPool(ckOffset + ik, 3);
              const double e = cMatPool(ckOffset + ik, 4);
              const double f = cMatPool(ckOffset + ik, 5);
              const double g = cMatPool(ckOffset + ik, 6);
              const double h = cMatPool(ckOffset + ik, 7);
              const double i = cMatPool(ckOffset + ik, 8);

              const double det = a * (e * i - f * h) - b * (d * i - f * g) +
                                 c * (d * h - e * g);

              ckInvMatPool(rank, 0) = (e * i - f * h) / det;
              ckInvMatPool(rank, 1) = -(b * i - c * h) / det;
              ckInvMatPool(rank, 2) = (b * f - c * e) / det;
              ckInvMatPool(rank, 3) = -(d * i - f * g) / det;
              ckInvMatPool(rank, 4) = (a * i - c * g) / det;
              ckInvMatPool(rank, 5) = -(a * f - c * d) / det;
              ckInvMatPool(rank, 6) = (d * h - e * g) / det;
              ckInvMatPool(rank, 7) = -(a * h - b * g) / det;
              ckInvMatPool(rank, 8) = (a * e - b * d) / det;
            });
          });
      Kokkos::fence();

      Kokkos::parallel_for(
          Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(workNodeSize,
                                                            Kokkos::AUTO()),
          KOKKOS_LAMBDA(
              const Kokkos::TeamPolicy<
                  Kokkos::DefaultExecutionSpace>::member_type &teamMember) {
            const int rank = teamMember.league_rank();

            // Get the I node indices and work size for this batch
            const int nodeI = mFarMatI(workingNode(rank));
            const int indexIStart = mClusterTree(nodeI, 2);
            const int indexIEnd = mClusterTree(nodeI, 3);

            const int rowSize = indexIEnd - indexIStart;

            const int ckOffset =
                workingNodeCMatOffset(rank, workingNodeIteration(rank));

            // For each row in the current C matrix block, update with the
            // product of the previously selected C (ckMatPool) and its inverse
            // (ckInvMatPool)
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(teamMember, rowSize), [&](const int j) {
                  const int index = ckOffset + j;
                  const int ckIndex = relativeCoordOffset(rank) + j;

                  // Initialize the 3x3 block to zero before accumulating the
                  // sum
                  for (int k = 0; k < 9; k++)
                    cMatPool(index, k) = 0.0;

                  double sum;
                  // Loop to perform matrix multiplication: cMat = ckMat * ckInv
                  for (int row = 0; row < 3; row++)
                    for (int col = 0; col < 3; col++) {
                      sum = 0.0;
                      for (int k = 0; k < 3; k++)
                        sum += ckMatPool(ckIndex, 3 * row + k) *
                               ckInvMatPool(rank, 3 * k + col);
                      cMatPool(index, 3 * row + col) = sum;
                    }
                });
          });
      Kokkos::fence();

      // Record the duration for C matrix normalization
      auto end = std::chrono::steady_clock::now();
      ckNormalizationDuration +=
          std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
              .count();
    }

    // Compute relative coordinates for Q matrix (QMat) inference
    {
      std::chrono::steady_clock::time_point begin =
          std::chrono::steady_clock::now();

      totalCoord = 0;
      Kokkos::parallel_scan(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workNodeSize),
          KOKKOS_LAMBDA(const int i, int &update, const bool final) {
            if (final)
              relativeCoordOffset(i) = update;
            const int nodeJ = mFarMatJ(workingNode(i));

            const int indexJStart = mClusterTree(nodeJ, 2);
            const int indexJEnd = mClusterTree(nodeJ, 3);

            const int workSizeJ = indexJEnd - indexJStart;

            update += workSizeJ;
          },
          totalCoord);
      totalNumQuery += totalCoord;

      // calculate the relative coordinates
      Kokkos::parallel_for(
          Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(workNodeSize,
                                                            Kokkos::AUTO()),
          KOKKOS_LAMBDA(
              const Kokkos::TeamPolicy<
                  Kokkos::DefaultExecutionSpace>::member_type &teamMember) {
            const int rank = teamMember.league_rank();
            const int nodeJ = mFarMatJ(workingNode(rank));
            const int relativeOffset = relativeCoordOffset(rank);

            const int indexJStart = mClusterTree(nodeJ, 2);
            const int indexJEnd = mClusterTree(nodeJ, 3);
            const int workSizeJ = indexJEnd - indexJStart;

            // Get the I node for this batch and the selected row
            const int nodeI = mFarMatI(workingNode(rank));
            const int indexI =
                mClusterTree(nodeI, 2) +
                workingNodeSelectedRowIdx(rank, workingNodeIteration(rank));

            Kokkos::parallel_for(Kokkos::TeamVectorRange(teamMember, workSizeJ),
                                 [&](const int j) {
                                   const int index = relativeOffset + j;
                                   for (int l = 0; l < 3; l++) {
                                     relativeCoordPool(3 * index + l) =
                                         -mCoord(indexI, l) +
                                         mCoord(indexJStart + j, l);
                                   }
                                 });
          });
      Kokkos::fence();

      // do inference for QMat
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

      auto resultTensor = mTwoBodyModel.forward(inputs).toTensor();

      // copy result to QMat
      auto dataPtr = resultTensor.data_ptr<float>();

      // Copy Q matrix predictions into qMatPool, enforcing symmetry for
      // off-diagonal elements by averaging with their transpose.
      Kokkos::parallel_for(
          Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(workNodeSize,
                                                            Kokkos::AUTO()),
          KOKKOS_LAMBDA(
              const Kokkos::TeamPolicy<
                  Kokkos::DefaultExecutionSpace>::member_type &teamMember) {
            const int rank = teamMember.league_rank();
            const int nodeJ = mFarMatJ(workingNode(rank));
            const int relativeOffset = relativeCoordOffset(rank);

            const int indexJStart = mClusterTree(nodeJ, 2);
            const int indexJEnd = mClusterTree(nodeJ, 3);
            const int workSizeJ = indexJEnd - indexJStart;

            const int qkOffset =
                workingNodeQMatOffset(rank, workingNodeIteration(rank));

            // For each row (target) in this batch, fill qMatPool
            Kokkos::parallel_for(Kokkos::TeamVectorRange(teamMember, workSizeJ),
                                 [&](const int j) {
                                   const int index = relativeOffset + j;

                                   for (int row = 0; row < 3; row++)
                                     for (int col = 0; col < 3; col++)
                                       if (row == col)
                                         qMatPool(qkOffset + j, 3 * row + col) =
                                             dataPtr[index * 9 + 3 * row + col];
                                       else {
                                         const int l1 = 3 * row + col;
                                         const int l2 = 3 * col + row;
                                         qMatPool(qkOffset + j, 3 * row + col) =
                                             0.5 * (dataPtr[index * 9 + l1] +
                                                    dataPtr[index * 9 + l2]);
                                       }
                                 });
          });
      Kokkos::fence();

      // Record query duration for QMat inference
      std::chrono::steady_clock::time_point end =
          std::chrono::steady_clock::now();
      queryDuration +=
          std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
              .count();
    }

    // find index for CMat
    {
      auto begin = std::chrono::steady_clock::now();

      // Orthogonalize Q matrices against previous basis and remove projections
      Kokkos::parallel_for(
          Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(workNodeSize,
                                                            Kokkos::AUTO()),
          KOKKOS_LAMBDA(
              const Kokkos::TeamPolicy<
                  Kokkos::DefaultExecutionSpace>::member_type &teamMember) {
            const int rank = teamMember.league_rank();

            // Get J node indices and column size
            const int nodeJ = mFarMatJ(workingNode(rank));
            const int indexJStart = mClusterTree(nodeJ, 2);
            const int indexJEnd = mClusterTree(nodeJ, 3);

            const int colSize = indexJEnd - indexJStart;

            const int qkOffset =
                workingNodeQMatOffset(rank, workingNodeIteration(rank));
            const int ik =
                workingNodeSelectedRowIdx(rank, workingNodeIteration(rank));

            // For each entry in the Q matrix block (flattened), subtract
            // projections
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(teamMember, colSize * 9),
                [&](const int j) {
                  const int index = qkOffset + j / 9;
                  const int k = j % 9;
                  const int row = k / 3;
                  const int col = k % 3;

                  double sum = 0.0;

                  // Loop over all previous iterations to project out old basis
                  for (int l = 0; l < innerNumIter - 1; l++) {
                    const int indexL = workingNodeQMatOffset(rank, l) + j / 9;
                    const int cMatOffsetIk =
                        workingNodeCMatOffset(rank, l) + ik;
                    for (int m = 0; m < 3; m++)
                      sum += cMatPool(cMatOffsetIk, 3 * row + m) *
                             qMatPool(indexL, 3 * m + col);
                  }
                  // Remove projection from Q
                  qMatPool(index, k) -= sum;
                });
          });
      Kokkos::fence();

      // Select the column (basis vector) with largest determinant (norm) as the
      // next basis for Q
      Kokkos::parallel_for(
          Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(workNodeSize,
                                                            Kokkos::AUTO()),
          KOKKOS_LAMBDA(
              const Kokkos::TeamPolicy<
                  Kokkos::DefaultExecutionSpace>::member_type &teamMember) {
            const int rank = teamMember.league_rank();

            // Get J node indices and column size
            const int nodeJ = mFarMatJ(workingNode(rank));
            const int indexJStart = mClusterTree(nodeJ, 2);
            const int indexJEnd = mClusterTree(nodeJ, 3);
            const int colSize = indexJEnd - indexJStart;

            const int qkOffset =
                workingNodeQMatOffset(rank, workingNodeIteration(rank));

            // Find the row in Q with the largest determinant, and not yet used
            Kokkos::MaxLoc<double, int>::value_type result;
            Kokkos::parallel_reduce(
                Kokkos::TeamVectorRange(teamMember, colSize),
                [&](const int j,
                    Kokkos::MaxLoc<double, int>::value_type &update) {
                  double curNorm = 0.0;
                  const int index = qkOffset + j;

                  // Compute determinant of the 3x3 Q block
                  const double a = qMatPool(index, 0);
                  const double b = qMatPool(index, 1);
                  const double c = qMatPool(index, 2);
                  const double d = qMatPool(index, 3);
                  const double e = qMatPool(index, 4);
                  const double f = qMatPool(index, 5);
                  const double g = qMatPool(index, 6);
                  const double h = qMatPool(index, 7);
                  const double i = qMatPool(index, 8);

                  curNorm = a * (e * i - f * h) - b * (d * i - f * g) +
                            c * (d * h - e * g);

                  // Check if this row has been selected before
                  bool exist = false;
                  for (int l = 0; l <= workingNodeIteration(rank); l++)
                    if (workingNodeSelectedColIdx(rank, l) == j)
                      exist = true;

                  // If not previously used and norm is greater, update
                  if (curNorm > update.val && exist == false) {
                    update.val = curNorm;
                    update.loc = j;
                  }
                },
                Kokkos::MaxLoc<double, int>(result));
            teamMember.team_barrier();

            workingNodeSelectedColIdx(rank, workingNodeIteration(rank) + 1) =
                result.loc;
          });
      Kokkos::fence();

      // Track Q matrix normalization duration
      auto end = std::chrono::steady_clock::now();
      qkNormalizationDuration +=
          std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
              .count();
    }

    // stop criterion
    {
      auto start = std::chrono::steady_clock::now();

      // For each batch, evaluate norms and cross-terms to update nu2 and mu2,
      // and check stopping condition
      Kokkos::parallel_for(
          Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(workNodeSize,
                                                            Kokkos::AUTO()),
          KOKKOS_LAMBDA(
              const Kokkos::TeamPolicy<
                  Kokkos::DefaultExecutionSpace>::member_type &teamMember) {
            const int rank = teamMember.league_rank();

            // Offsets for the current iteration's C and Q blocks
            const int ckOffset =
                workingNodeCMatOffset(rank, workingNodeIteration(rank));
            const int qkOffset =
                workingNodeQMatOffset(rank, workingNodeIteration(rank));

            // Indices for I and J, and work sizes
            const int nodeI = mFarMatI(workingNode(rank));
            const int indexIStart = mClusterTree(nodeI, 2);
            const int indexIEnd = mClusterTree(nodeI, 3);
            const int rowSize = indexIEnd - indexIStart;

            const int nodeJ = mFarMatJ(workingNode(rank));
            const int indexJStart = mClusterTree(nodeJ, 2);
            const int indexJEnd = mClusterTree(nodeJ, 3);
            const int colSize = indexJEnd - indexJStart;

            ArrReduce ckArrReduce, qkArrReduce;

            // Accumulate squared norms and dot products for the current C
            // matrix (row-wise)
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(teamMember, rowSize * 6),
                [&](const int j, ArrReduce &tSum) {
                  const int row = j / 6;
                  const int k = j % 6;

                  if (k < 3)
                    // Sum of squared values for each column
                    tSum.values[k] += cMatPool(ckOffset + row, k) *
                                          cMatPool(ckOffset + row, k) +
                                      cMatPool(ckOffset + row, k + 3) *
                                          cMatPool(ckOffset + row, k + 3) +
                                      cMatPool(ckOffset + row, k + 6) *
                                          cMatPool(ckOffset + row, k + 6);
                  else
                    // Off-diagonal dot products between columns
                    tSum.values[k] +=
                        cMatPool(ckOffset + row, k % 3) *
                            cMatPool(ckOffset + row, (k + 1) % 3) +
                        cMatPool(ckOffset + row, k % 3 + 3) *
                            cMatPool(ckOffset + row, (k + 1) % 3 + 3) +
                        cMatPool(ckOffset + row, k % 3 + 6) *
                            cMatPool(ckOffset + row, (k + 1) % 3 + 6);
                },
                Kokkos::Sum<ArrReduce>(ckArrReduce));

            // Accumulate squared norms and dot products for the current Q
            // matrix (col-wise)
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(teamMember, colSize),
                [&](const int j, ArrReduce &tSum) {
                  tSum.values[0] +=
                      qMatPool(qkOffset + j, 0) * qMatPool(qkOffset + j, 0) +
                      qMatPool(qkOffset + j, 1) * qMatPool(qkOffset + j, 1) +
                      qMatPool(qkOffset + j, 2) * qMatPool(qkOffset + j, 2);

                  tSum.values[1] +=
                      qMatPool(qkOffset + j, 3) * qMatPool(qkOffset + j, 3) +
                      qMatPool(qkOffset + j, 4) * qMatPool(qkOffset + j, 4) +
                      qMatPool(qkOffset + j, 5) * qMatPool(qkOffset + j, 5);

                  tSum.values[2] +=
                      qMatPool(qkOffset + j, 6) * qMatPool(qkOffset + j, 6) +
                      qMatPool(qkOffset + j, 7) * qMatPool(qkOffset + j, 7) +
                      qMatPool(qkOffset + j, 8) * qMatPool(qkOffset + j, 8);

                  tSum.values[3] +=
                      qMatPool(qkOffset + j, 0) * qMatPool(qkOffset + j, 3) +
                      qMatPool(qkOffset + j, 1) * qMatPool(qkOffset + j, 4) +
                      qMatPool(qkOffset + j, 2) * qMatPool(qkOffset + j, 5);

                  tSum.values[4] +=
                      qMatPool(qkOffset + j, 3) * qMatPool(qkOffset + j, 6) +
                      qMatPool(qkOffset + j, 4) * qMatPool(qkOffset + j, 7) +
                      qMatPool(qkOffset + j, 5) * qMatPool(qkOffset + j, 8);

                  tSum.values[5] +=
                      qMatPool(qkOffset + j, 6) * qMatPool(qkOffset + j, 0) +
                      qMatPool(qkOffset + j, 7) * qMatPool(qkOffset + j, 1) +
                      qMatPool(qkOffset + j, 8) * qMatPool(qkOffset + j, 2);
                },
                Kokkos::Sum<ArrReduce>(qkArrReduce));

            teamMember.team_barrier();

            // Compute nu2 and mu2 for the current node/batch
            Kokkos::single(Kokkos::PerTeam(teamMember), [&]() {
              const double ckCol0SquaredNorm = ckArrReduce.values[0];
              const double ckCol1SquaredNorm = ckArrReduce.values[1];
              const double ckCol2SquaredNorm = ckArrReduce.values[2];

              const double ck01Dot = ckArrReduce.values[3];
              const double ck12Dot = ckArrReduce.values[4];
              const double ck20Dot = ckArrReduce.values[5];

              const double qkRow0SquaredNorm = qkArrReduce.values[0];
              const double qkRow1SquaredNorm = qkArrReduce.values[1];
              const double qkRow2SquaredNorm = qkArrReduce.values[2];

              const double qk01Dot = qkArrReduce.values[3];
              const double qk12Dot = qkArrReduce.values[4];
              const double qk20Dot = qkArrReduce.values[5];

              nu2(rank) = ckCol0SquaredNorm * qkRow0SquaredNorm +
                          ckCol1SquaredNorm * qkRow1SquaredNorm +
                          ckCol2SquaredNorm * qkRow2SquaredNorm +
                          2.0 * qk01Dot * ck01Dot + 2.0 * qk12Dot * ck12Dot +
                          2.0 * qk20Dot * ck20Dot;

              mu2(rank) += nu2(rank);
            });
            teamMember.team_barrier();

            for (int l = 0; l < workingNodeIteration(rank) - 1; l++) {
              const int ckLOffset = workingNodeCMatOffset(rank, l);
              const int qkLOffset = workingNodeQMatOffset(rank, l);

              for (int d1 = 0; d1 < 3; d1++)
                for (int d2 = 0; d2 < 3; d2++) {
                  double ckDot = 0.0, qkDot = 0.0;
                  // Compute dot product for the C basis
                  Kokkos::parallel_reduce(
                      Kokkos::TeamThreadRange(teamMember, rowSize),
                      [&](const int j, double &tSum) {
                        tSum += cMatPool(ckOffset + j, d1) *
                                    cMatPool(ckLOffset + j, d2) +
                                cMatPool(ckOffset + j, d1 + 3) *
                                    cMatPool(ckLOffset + j, d2 + 3) +
                                cMatPool(ckOffset + j, d1 + 6) *
                                    cMatPool(ckLOffset + j, d2 + 6);
                      },
                      Kokkos::Sum<double>(ckDot));

                  // Compute dot product for the Q basis
                  Kokkos::parallel_reduce(
                      Kokkos::TeamThreadRange(teamMember, colSize),
                      [&](const int j, double &tSum) {
                        tSum += qMatPool(qkOffset + j, 3 * d1) *
                                    qMatPool(qkLOffset + j, 3 * d2) +
                                qMatPool(qkOffset + j, 3 * d1 + 1) *
                                    qMatPool(qkLOffset + j, 3 * d2 + 1) +
                                qMatPool(qkOffset + j, 3 * d1 + 2) *
                                    qMatPool(qkLOffset + j, 3 * d2 + 2);
                      },
                      Kokkos::Sum<double>(qkDot));

                  if (teamMember.team_rank() == 0)
                    mu2(rank) += 2.0 * abs(ckDot) * abs(qkDot);
                  teamMember.team_barrier();
                }
            }
          });
      Kokkos::fence();

      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workNodeSize),
          KOKKOS_LAMBDA(const int i) {
            // Retrieve the I node and compute the row (work) size for this
            // node.
            const int nodeI = mFarMatI(workingNode(i));
            const int indexIStart = mClusterTree(nodeI, 2);
            const int indexIEnd = mClusterTree(nodeI, 3);
            const int rowSize = indexIEnd - indexIStart;

            // Retrieve the J node and compute the column size for this node.
            const int nodeJ = mFarMatJ(workingNode(i));
            const int indexJStart = mClusterTree(nodeJ, 2);
            const int indexJEnd = mClusterTree(nodeJ, 3);
            const int colSize = indexJEnd - indexJStart;

            // Advance the iteration counter for this work node.
            workingNodeIteration(i)++;

            // Check stopping criterion for this batch/node.
            if (nu2(i) < mu2(i) * epsilon2 ||
                workingNodeIteration(i) >= maxIter ||
                workingNodeIteration(i) >= std::min(rowSize, colSize)) {
              stopNode(i) = -1;
            } else {
              stopNode(i) = 0;
            }
          });
      Kokkos::fence();

      {
        int iterationCheckResult = 0;

        // Check if any node exceeded the maximum allowed number of iterations.
        Kokkos::parallel_reduce(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workNodeSize),
            KOKKOS_LAMBDA(const int i, int &tIterationCheckResult) {
              if (workingNodeIteration(i) > maxIter)
                tIterationCheckResult++;
            },
            Kokkos::Sum<int>(iterationCheckResult));
        Kokkos::fence();

        if (iterationCheckResult > 0) {
          // If any node has excessive iterations, print debug information.
          std::cout << "Number of iteration is large in rank: " << mMPIRank
                    << std::endl;

          DeviceIndexVector::HostMirror farMatIHost =
              Kokkos::create_mirror_view(mFarMatI);
          DeviceIndexVector::HostMirror farMatJHost =
              Kokkos::create_mirror_view(mFarMatJ);
          Kokkos::deep_copy(farMatIHost, mFarMatI);
          Kokkos::deep_copy(farMatJHost, mFarMatJ);

          DeviceIndexMatrix::HostMirror clusterTreeHost =
              Kokkos::create_mirror_view(mClusterTree);
          Kokkos::deep_copy(clusterTreeHost, mClusterTree);

          DeviceIntVector::HostMirror workingNodeIterationHost =
              Kokkos::create_mirror_view(workingNodeIteration);
          Kokkos::deep_copy(workingNodeIterationHost, workingNodeIteration);

          DeviceIntVector::HostMirror workingNodeHost =
              Kokkos::create_mirror_view(workingNode);
          Kokkos::deep_copy(workingNodeHost, workingNode);

          DeviceDoubleVector::HostMirror nuHost =
              Kokkos::create_mirror_view(nu2);
          DeviceDoubleVector::HostMirror muHost =
              Kokkos::create_mirror_view(mu2);
          Kokkos::deep_copy(nuHost, nu2);
          Kokkos::deep_copy(muHost, mu2);

          // Print detailed iteration and node information for diagnosis.
          for (int i = 0; i < workNodeSize; i++) {
            if (workingNodeIterationHost(i) > maxIter) {
              std::cout << "farMatI: " << farMatIHost(workingNodeHost(i))
                        << " row size: "
                        << clusterTreeHost(farMatIHost(workingNodeHost(i)), 3) -
                               clusterTreeHost(farMatIHost(workingNodeHost(i)),
                                               2)
                        << " farMatJ: " << farMatJHost(workingNodeHost(i))
                        << " col size: "
                        << clusterTreeHost(farMatJHost(workingNodeHost(i)), 3) -
                               clusterTreeHost(farMatJHost(workingNodeHost(i)),
                                               2)
                        << " mu: " << muHost(i) << " nu: " << nuHost(i)
                        << std::endl;
            }
          }
        }
      }

      auto end = std::chrono::steady_clock::now();
      stopCriterionDuration +=
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count();
    }

    // Print detailed iteration and node information for diagnosis.
    int newWorkNodeSize = 0;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workNodeSize),
        KOKKOS_LAMBDA(const std::size_t i, int &tNewWorkNodeSize) {
          if (stopNode(i) != -1) {
            tNewWorkNodeSize += 1;
          }
        },
        Kokkos::Sum<int>(newWorkNodeSize));

    // dot product
    {
      auto start = std::chrono::steady_clock::now();

      // Stage 1: Identify nodes that have converged (stopNode == -1) for dot
      // product.
      const int dotSize = workNodeSize - newWorkNodeSize;
      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, dotSize),
          KOKKOS_LAMBDA(const int rank) {
            int counter = 0;
            for (int i = 0; i < workNodeSize; i++) {
              if (stopNode(i) == -1) {
                if (counter == rank) {
                  dotProductNode(rank) = workingNode(i);
                  dotProductRank(rank) = i;
                  break;
                }
                counter++;
              }
            }
          });
      Kokkos::fence();

      // Initialize the middle matrix pool to zero for accumulation in the next
      // step.
      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
              0, innerNumIter * dotSize * 3),
          KOKKOS_LAMBDA(const int i) { middleMatPool(i) = 0.0; });
      Kokkos::fence();

      Kokkos::parallel_for(
          Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(dotSize,
                                                            Kokkos::AUTO()),
          KOKKOS_LAMBDA(
              const Kokkos::TeamPolicy<
                  Kokkos::DefaultExecutionSpace>::member_type &teamMember) {
            const int rank = teamMember.league_rank();
            const int workingNodeRank = dotProductRank(rank);

            const int nodeJ = mFarMatJ(workingNode(workingNodeRank));
            const int indexJStart = mClusterTree(nodeJ, 2);
            const int indexJEnd = mClusterTree(nodeJ, 3);
            const int workSizeJ = indexJEnd - indexJStart;

            Kokkos::parallel_for(
                Kokkos::TeamThreadMDRange(teamMember, innerNumIter * 3,
                                          workSizeJ),
                [&](const int i, const int j) {
                  const int iter = i / 3;
                  const int row = i % 3;

                  const int qkOffset =
                      workingNodeQMatOffset(workingNodeRank, iter);
                  const int middleMatOffset = 3 * innerNumIter * rank;

                  double sum = 0.0;
                  for (int k = 0; k < 3; k++)
                    sum += qMatPool(qkOffset + j, row * 3 + k) *
                           f(indexJStart + j, k);
                  Kokkos::single(Kokkos::PerThread(teamMember), [&]() {
                    Kokkos::atomic_add(&middleMatPool(middleMatOffset + i),
                                       sum);
                  });
                });
          });
      Kokkos::fence();

      Kokkos::parallel_for(
          Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(dotSize,
                                                            Kokkos::AUTO()),
          KOKKOS_LAMBDA(
              const Kokkos::TeamPolicy<
                  Kokkos::DefaultExecutionSpace>::member_type &teamMember) {
            const int rank = teamMember.league_rank();
            const int workingNodeRank = dotProductRank(rank);

            // Get node I information (rows of solution u)
            const int nodeI = mFarMatI(dotProductNode(rank));
            const int indexIStart = mClusterTree(nodeI, 2);
            const int indexIEnd = mClusterTree(nodeI, 3);
            const int workSizeI = indexIEnd - indexIStart;

            // For each row, for each component, and for each inner iteration,
            // accumulate contributions to u
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(teamMember,
                                        workSizeI * innerNumIter * 3),
                [&](const int i) {
                  const int index = i / (3 * innerNumIter);
                  const int row = (i % (3 * innerNumIter)) / innerNumIter;
                  const int iter = (i % (3 * innerNumIter)) % innerNumIter;

                  double sum = 0.0;
                  const int cMatOffset =
                      workingNodeCMatOffset(workingNodeRank, iter);
                  const int middleMatOffset = 3 * innerNumIter * rank;

                  // Matrix-vector multiply: sum C * middleMat over 'k'
                  for (int k = 0; k < 3; k++)
                    sum += cMatPool(cMatOffset + index, row * 3 + k) *
                           middleMatPool(middleMatOffset + 3 * iter + k);

                  Kokkos::atomic_add(&u(indexIStart + index, row), sum);
                });
          });
      Kokkos::fence();

      // Stage 2, consider the symmetry property
      if (mUseSymmetry) {
        Kokkos::parallel_for(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                0, innerNumIter * dotSize * 3),
            KOKKOS_LAMBDA(const int i) { middleMatPool(i) = 0.0; });
        Kokkos::fence();

        // Compute (C^T * f) for each row, exploiting symmetry
        Kokkos::parallel_for(
            Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(dotSize,
                                                              Kokkos::AUTO()),
            KOKKOS_LAMBDA(
                const Kokkos::TeamPolicy<
                    Kokkos::DefaultExecutionSpace>::member_type &teamMember) {
              const int rank = teamMember.league_rank();
              const int workingNodeRank = dotProductRank(rank);

              // Get node I info (rows)
              const int nodeI = mFarMatI(workingNode(workingNodeRank));
              const int indexIStart = mClusterTree(nodeI, 2);
              const int indexIEnd = mClusterTree(nodeI, 3);
              const int workSizeI = indexIEnd - indexIStart;

              Kokkos::parallel_for(
                  Kokkos::TeamThreadMDRange(teamMember, innerNumIter * 3,
                                            workSizeI),
                  [&](const int i, const int j) {
                    const int iter = i / 3;
                    const int row = i % 3;

                    const int ckOffset =
                        workingNodeCMatOffset(workingNodeRank, iter);
                    const int middleMatOffset = 3 * innerNumIter * rank;

                    double sum = 0.0;
                    for (int k = 0; k < 3; k++)
                      sum += cMatPool(ckOffset + j, k * 3 + row) *
                             f(indexIStart + j, k);
                    Kokkos::single(Kokkos::PerThread(teamMember), [&]() {
                      Kokkos::atomic_add(&middleMatPool(middleMatOffset + i),
                                         sum);
                    });
                  });
            });
        Kokkos::fence();

        // Accumulate Q^T * (result of previous step) for J node columns
        Kokkos::parallel_for(
            Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(dotSize,
                                                              Kokkos::AUTO()),
            KOKKOS_LAMBDA(
                const Kokkos::TeamPolicy<
                    Kokkos::DefaultExecutionSpace>::member_type &teamMember) {
              const int rank = teamMember.league_rank();
              const int workingNodeRank = dotProductRank(rank);

              // Get node J info (columns)
              const int nodeJ = mFarMatJ(dotProductNode(rank));
              const int indexJStart = mClusterTree(nodeJ, 2);
              const int indexJEnd = mClusterTree(nodeJ, 3);
              const int workSizeJ = indexJEnd - indexJStart;

              // Flat iteration over (column, component, inner iteration)
              Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(teamMember,
                                          workSizeJ * innerNumIter * 3),
                  [&](const int i) {
                    const int index = i / (3 * innerNumIter);
                    const int row = (i % (3 * innerNumIter)) / innerNumIter;
                    const int iter = (i % (3 * innerNumIter)) % innerNumIter;

                    double sum = 0.0;
                    const int qMatOffset =
                        workingNodeQMatOffset(workingNodeRank, iter);
                    const int middleMatOffset = 3 * innerNumIter * rank;

                    for (int k = 0; k < 3; k++)
                      sum += qMatPool(qMatOffset + index, k * 3 + row) *
                             middleMatPool(middleMatOffset + 3 * iter + k);

                    // Atomic add to final solution vector
                    Kokkos::atomic_add(&u(indexJStart + index, row), sum);
                  });
            });
        Kokkos::fence();
      }

      // post check
      if (postCheck) {
        // Reset the uDotCheck flag for all working nodes
        Kokkos::parallel_for(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, dotSize),
            KOKKOS_LAMBDA(const int rank) { uDotCheck(rank) = 0; });
        Kokkos::fence();

        // For each working node, compute the Euclidean norm of the u solution
        // vector for each row
        Kokkos::parallel_for(
            Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(dotSize,
                                                              Kokkos::AUTO()),
            KOKKOS_LAMBDA(
                const Kokkos::TeamPolicy<
                    Kokkos::DefaultExecutionSpace>::member_type &teamMember) {
              const int rank = teamMember.league_rank();

              // Get row range for this working node
              const int nodeI = mFarMatI(dotProductNode(rank));
              const int indexIStart = mClusterTree(nodeI, 2);
              const int indexIEnd = mClusterTree(nodeI, 3);
              const int workSizeI = indexIEnd - indexIStart;

              // For each row, compute norm of 3-vector [u_x, u_y, u_z]
              Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(teamMember, workSizeI),
                  [&](const int i) {
                    double uNorm = 0.0;
                    for (int j = 0; j < 3; j++)
                      uNorm += pow(u(indexIStart + i, j), 2);
                    uNorm = sqrt(uNorm);
                    if (uNorm > 1e6)
                      uDotCheck(rank) = 1;
                  });
            });
        Kokkos::fence();

        // Reduce uDotCheck array to count how many ranks failed the check
        int dotCheckSum = 0;
        Kokkos::parallel_reduce(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, dotSize),
            KOKKOS_LAMBDA(const int rank, int &tDotCheck) {
              if (uDotCheck(rank) == 1) {
                tDotCheck += 1;
              }
            },
            Kokkos::Sum<int>(dotCheckSum));

        if (dotCheckSum > 0) {
          std::cout << "uNorm is too large in rank: " << mMPIRank << std::endl;

          // Create host mirror views for device data (for debugging/inspection
          // on the host)
          DeviceIndexVector::HostMirror farMatIHost =
              Kokkos::create_mirror_view(mFarMatI);
          DeviceIndexVector::HostMirror farMatJHost =
              Kokkos::create_mirror_view(mFarMatJ);
          Kokkos::deep_copy(farMatIHost, mFarMatI);
          Kokkos::deep_copy(farMatJHost, mFarMatJ);

          DeviceIndexMatrix::HostMirror clusterTreeHost =
              Kokkos::create_mirror_view(mClusterTree);
          Kokkos::deep_copy(clusterTreeHost, mClusterTree);

          DeviceIntVector::HostMirror dotCheckHost =
              Kokkos::create_mirror_view(uDotCheck);
          Kokkos::deep_copy(dotCheckHost, uDotCheck);

          DeviceIntVector::HostMirror dotProductNodeHost =
              Kokkos::create_mirror_view(dotProductNode);
          Kokkos::deep_copy(dotProductNodeHost, dotProductNode);

          // Loop through all dot-checked nodes and print details for those with
          // too-large norms
          for (int i = 0; i < dotSize; i++) {
            if (dotCheckHost(i) == 1) {
              std::cout
                  << "farMatI: " << farMatIHost(dotProductNodeHost(i))
                  << " row size: "
                  << clusterTreeHost(farMatIHost(dotProductNodeHost(i)), 3) -
                         clusterTreeHost(farMatIHost(dotProductNodeHost(i)), 2)
                  << " farMatJ: " << farMatJHost(dotProductNodeHost(i))
                  << " col size: "
                  << clusterTreeHost(farMatJHost(dotProductNodeHost(i)), 3) -
                         clusterTreeHost(farMatJHost(dotProductNodeHost(i)), 2)
                  << std::endl;
            }
          }
        }
      }

      auto end = std::chrono::steady_clock::now();
      dotDuration +=
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count();
    }

    // reset working node
    if (newWorkNodeSize == 0) {
      finishedNodeSize += workNodeSize;
      workNodeSize = 0;

      // Track the maximum number of inner iterations for reporting.
      if (maxInnerNumIter < innerNumIter) {
        maxInnerNumIter = innerNumIter;
      }

      innerNumIter = 0;
      // If only some nodes remain for further iteration, compact the arrays
      // and remap node indices.
    } else if (newWorkNodeSize < workNodeSize) {
      auto start = std::chrono::steady_clock::now();

      finishedNodeSize += workNodeSize - newWorkNodeSize;
      // copy working node arrays
      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workNodeSize),
          KOKKOS_LAMBDA(const int rank) {
            workingNodeCopy(rank) = workingNode(rank);
            int counter = rank + 1;
            if (rank < newWorkNodeSize)
              for (int i = 0; i < workNodeSize; i++) {
                if (stopNode(i) != -1) {
                  counter--;
                }
                if (counter == 0) {
                  workingNodeCopyOffset(rank) = i;
                  break;
                }
              }
          });
      Kokkos::fence();

      // Overwrite the workingNode array with only the remaining (active)
      // nodes.
      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,
                                                             newWorkNodeSize),
          KOKKOS_LAMBDA(const int rank) {
            workingNode(rank) = workingNodeCopy(workingNodeCopyOffset(rank));
          });
      Kokkos::fence();

      // For each of the following arrays, compact and reindex them in
      // accordance with the new reduced set of nodes:

      // Compact and remap C matrix offsets.
      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
              0, workNodeSize * maxIter),
          KOKKOS_LAMBDA(const int i) {
            workingNodeCopy(i) =
                workingNodeCMatOffset(i / maxIter, i % maxIter);
          });
      Kokkos::fence();

      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
              0, newWorkNodeSize * maxIter),
          KOKKOS_LAMBDA(const int i) {
            workingNodeCMatOffset(i / maxIter, i % maxIter) = workingNodeCopy(
                workingNodeCopyOffset(i / maxIter) * maxIter + i % maxIter);
          });
      Kokkos::fence();

      // Compact and remap Q matrix offsets.
      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
              0, workNodeSize * maxIter),
          KOKKOS_LAMBDA(const int i) {
            workingNodeCopy(i) =
                workingNodeQMatOffset(i / maxIter, i % maxIter);
          });
      Kokkos::fence();

      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
              0, newWorkNodeSize * maxIter),
          KOKKOS_LAMBDA(const int i) {
            workingNodeQMatOffset(i / maxIter, i % maxIter) = workingNodeCopy(
                workingNodeCopyOffset(i / maxIter) * maxIter + i % maxIter);
          });
      Kokkos::fence();

      // selected col
      // Compact and remap the selected column indices for each inner
      // iteration.
      for (int i = 0; i <= innerNumIter; i++) {
        Kokkos::parallel_for(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workNodeSize),
            KOKKOS_LAMBDA(const int rank) {
              workingNodeCopy(rank) = workingNodeSelectedColIdx(rank, i);
            });
        Kokkos::fence();

        Kokkos::parallel_for(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,
                                                               newWorkNodeSize),
            KOKKOS_LAMBDA(const int rank) {
              workingNodeSelectedColIdx(rank, i) =
                  workingNodeCopy(workingNodeCopyOffset(rank));
            });
        Kokkos::fence();
      }

      // selected row
      // Compact and remap the selected row indices for each inner iteration.
      for (int i = 0; i <= innerNumIter; i++) {
        Kokkos::parallel_for(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workNodeSize),
            KOKKOS_LAMBDA(const int rank) {
              workingNodeCopy(rank) = workingNodeSelectedRowIdx(rank, i);
            });
        Kokkos::fence();

        Kokkos::parallel_for(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,
                                                               newWorkNodeSize),
            KOKKOS_LAMBDA(const int rank) {
              workingNodeSelectedRowIdx(rank, i) =
                  workingNodeCopy(workingNodeCopyOffset(rank));
            });
        Kokkos::fence();
      }

      // num of iteration
      // Compact and remap the number of iterations for each working node.
      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workNodeSize),
          KOKKOS_LAMBDA(const int rank) {
            workingNodeCopy(rank) = workingNodeIteration(rank);
          });
      Kokkos::fence();

      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,
                                                             newWorkNodeSize),
          KOKKOS_LAMBDA(const int rank) {
            workingNodeIteration(rank) =
                workingNodeCopy(workingNodeCopyOffset(rank));
          });
      Kokkos::fence();

      // mu2
      // Compact and remap the mu2 tracking array (typically related to
      // error/norm criteria).
      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workNodeSize),
          KOKKOS_LAMBDA(const int rank) {
            workingNodeDoubleCopy(rank) = mu2(rank);
          });
      Kokkos::fence();

      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,
                                                             newWorkNodeSize),
          KOKKOS_LAMBDA(const int rank) {
            mu2(rank) = workingNodeDoubleCopy(workingNodeCopyOffset(rank));
          });
      Kokkos::fence();

      // Update the number of active work nodes.
      workNodeSize = newWorkNodeSize;

      auto end = std::chrono::steady_clock::now();
      resetDuration +=
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count();
    }

    if (finishedNodeSize == farNodeSize) {
      break;
    }
  }

  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

  MPI_Barrier(MPI_COMM_WORLD);

  // Perform reductions to aggregate statistics across all MPI ranks.
  MPI_Allreduce(MPI_IN_PLACE, &totalNumQuery, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &totalNumIter, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &maxInnerNumIter, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);

  // Print timing and summary statistics on rank 0 only
  if (mMPIRank == 0) {
    printf(
        "num query: %ld, num iteration: %ld, query duration: %.4fs, dot "
        "duration: %.4fs\n",
        totalNumQuery, totalNumIter, queryDuration / 1e6, dotDuration / 1e6);
    printf(
        "ck normalization duration: %.4fs, qk normalization duration: %.4fs, "
        "stop criterion duration: %.4fs, reset duration: %.4fs\n",
        ckNormalizationDuration / 1e6, qkNormalizationDuration / 1e6,
        stopCriterionDuration / 1e6, resetDuration / 1e6);
    printf("max inner num iter: %d\n", maxInnerNumIter);

    printf(
        "End of far dot. Dot time %.4fs\n",
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() /
            1e6);
  }
}