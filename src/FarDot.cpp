#include "HignnModel.hpp"

struct ArrReduce {
  double values[6];

  KOKKOS_INLINE_FUNCTION ArrReduce() {
    for (int i = 0; i < 6; i++) {
      values[i] = 0;
    }
  }

  KOKKOS_INLINE_FUNCTION ArrReduce(const ArrReduce &rhs) {
    for (int i = 0; i < 6; i++) {
      values[i] = rhs.values[i];
    }
  }

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

// Flag used for determinant check
#define FarDotDeterminantCheck

// Flag used for post check the velocity
#define FarDotVelocityCheck

void HignnModel::FarDot(DeviceDoubleMatrix u, DeviceDoubleMatrix f) {
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

  if (mMPIRank == 0)
    std::cout << "start of FarDot" << std::endl;
  MPI_Barrier(MPI_COMM_WORLD);

  double queryDuration = 0.0;
  double dotDuration = 0.0;
  double ckNormalizationDuration = 0.0;
  double qkNormalizationDuration = 0.0;
  double stopCriterionDuration = 0.0;
  double resetDuration = 0.0;

  std::size_t totalNumQuery = 0;
  std::size_t totalNumIter = 0;
  int innerNumIter = 0;
  int maxInnerNumIter = 0;

  const int maxRelativeCoord = mMaxRelativeCoord;
  const int matPoolSize = maxRelativeCoord * mMatPoolSizeFactor;
  const int maxWorkNodeSize = mMaxFarDotWorkNodeSize;
  const int maxIter = mMaxIter;
  const int middleMatPoolSize = maxWorkNodeSize * maxIter;
  int workNodeSize = 0;
  int allowedWorkload = 0;

  const bool postCheck = false;

  DeviceFloatVector relativeCoordPool("relativeCoordPool",
                                      maxRelativeCoord * 3);
  DeviceDoubleMatrix cMatPool("cMatPool", matPoolSize, 9);
  DeviceDoubleMatrix qMatPool("qMatPool", matPoolSize, 9);
  DeviceDoubleVector middleMatPool("middleMatPool", middleMatPoolSize * 3);
  DeviceDoubleMatrix ckMatPool("ckMatPool", maxRelativeCoord, 9);
  DeviceDoubleMatrix ckInvMatPool("ckInvMatPool", maxWorkNodeSize, 9);

  DeviceIntVector workingNode("workingNode", maxWorkNodeSize);
  DeviceIntVector dotProductNode("dotProductNode", maxWorkNodeSize);
  DeviceIntVector dotProductRank("dotProductRank", maxWorkNodeSize);
  DeviceIntVector stopNode("stopNode", maxWorkNodeSize);
  DeviceIntMatrix workingNodeCMatOffset("workingNodeCMatOffset",
                                        maxWorkNodeSize, maxIter);
  DeviceIntMatrix workingNodeQMatOffset("workingNodeQMatOffset",
                                        maxWorkNodeSize, maxIter);
  DeviceIntMatrix workingNodeSelectedColIdx("workingNodeSelectedColIdx",
                                            maxWorkNodeSize, maxIter);
  DeviceIntMatrix workingNodeSelectedRowIdx("workingNodeSelectedRowIdx",
                                            maxWorkNodeSize, maxIter);
  DeviceIntVector workingNodeIteration("workingNodeIteration", maxWorkNodeSize);

  DeviceIntVector workingNodeCopy("workingNodeCopy", maxWorkNodeSize * maxIter);
  DeviceIntVector workingNodeCopyOffset("workingNodeCopyOffset",
                                        maxWorkNodeSize);

  DeviceDoubleVector nu2("nu2", maxWorkNodeSize);
  DeviceDoubleVector mu2("mu", maxWorkNodeSize);

  DeviceDoubleVector workingNodeDoubleCopy("workingNodeDoubleCopy",
                                           maxWorkNodeSize);

  DeviceIntVector uDotCheck("uDotCheck", maxWorkNodeSize);

  const double epsilon = mEpsilon;
  const double epsilon2 = epsilon * epsilon;

  DeviceIntVector relativeCoordOffset("relativeCoordOffset", maxWorkNodeSize);

  auto &mFarMatI = *mFarMatIPtr;
  auto &mFarMatJ = *mFarMatJPtr;
  auto &mCoord = *mCoordPtr;
  auto &mClusterTree = *mClusterTreePtr;

  const int farNodeSize = mFarMatI.extent(0);
  int finishedNodeSize = 0;
  int installedNode = 0;
  int totalCoord = 0;

  bool farDotFinished = false;
  while (!farDotFinished) {
    totalNumIter++;
    innerNumIter++;

    // select working node
    if (workNodeSize == 0) {
      allowedWorkload = maxRelativeCoord;

      // estimate the workload
      int estimatedWorkload;
      int leftNode = std::min(farNodeSize - finishedNodeSize, maxWorkNodeSize);
      workNodeSize = leftNode;
      int lowerWorkNodeSize = 0;
      int upperWorkNodeSize = workNodeSize;
      // install new working node
      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workNodeSize),
          KOKKOS_LAMBDA(const int i) { workingNode(i) = installedNode + i; });
      while (true) {
        int estimatedQMatWorkload = 0;
        int estimatedCMatWorkload = 0;

        Kokkos::parallel_reduce(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workNodeSize),
            KOKKOS_LAMBDA(const std::size_t i, int &tSum) {
              int nodeI = mFarMatI(workingNode(i));
              int workload = mClusterTree(nodeI, 3) - mClusterTree(nodeI, 2);
              tSum += workload;
            },
            Kokkos::Sum<int>(estimatedCMatWorkload));

        Kokkos::parallel_reduce(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workNodeSize),
            KOKKOS_LAMBDA(const std::size_t i, int &tSum) {
              int nodeJ = mFarMatJ(workingNode(i));
              int workload = mClusterTree(nodeJ, 3) - mClusterTree(nodeJ, 2);
              tSum += workload;
            },
            Kokkos::Sum<int>(estimatedQMatWorkload));

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

            const int ckOffset =
                workingNodeCMatOffset(rank, workingNodeIteration(rank));
            const int jk =
                workingNodeSelectedColIdx(rank, workingNodeIteration(rank));

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

            Kokkos::MaxLoc<double, int>::value_type result;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(teamMember, rowSize),
                [&](const int j,
                    Kokkos::MaxLoc<double, int>::value_type &update) {
                  double curNorm = 0.0;
                  const int index = ckOffset + j;
                  const int ckIndex = relativeCoordOffset(rank) + j;

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

                  for (int k = 0; k < 9; k++)
                    ckMatPool(ckIndex, k) = cMatPool(index, k);

                  bool exist = false;
                  for (int l = 0; l < workingNodeIteration(rank); l++)
                    if (workingNodeSelectedRowIdx(rank, l) == j)
                      exist = true;

                  if (curNorm > update.val && exist == false) {
                    update.val = curNorm;
                    update.loc = j;
                  }
                },
                Kokkos::MaxLoc<double, int>(result));
            teamMember.team_barrier();

            workingNodeSelectedRowIdx(rank, workingNodeIteration(rank)) =
                result.loc;

            const int ik = result.loc;

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

            const int nodeI = mFarMatI(workingNode(rank));
            const int indexIStart = mClusterTree(nodeI, 2);
            const int indexIEnd = mClusterTree(nodeI, 3);

            const int rowSize = indexIEnd - indexIStart;

            const int ckOffset =
                workingNodeCMatOffset(rank, workingNodeIteration(rank));

            // multiply ckInv to cMat
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(teamMember, rowSize), [&](const int j) {
                  const int index = ckOffset + j;
                  const int ckIndex = relativeCoordOffset(rank) + j;

                  for (int k = 0; k < 9; k++)
                    cMatPool(index, k) = 0.0;

                  double sum;
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

      auto end = std::chrono::steady_clock::now();
      ckNormalizationDuration +=
          std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
              .count();
    }

    {
      std::chrono::steady_clock::time_point begin =
          std::chrono::steady_clock::now();

      // calculate relative coord for Q
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

      std::chrono::steady_clock::time_point end =
          std::chrono::steady_clock::now();
      queryDuration +=
          std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
              .count();
    }

    // find index for CMat
    {
      auto begin = std::chrono::steady_clock::now();
      Kokkos::parallel_for(
          Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(workNodeSize,
                                                            Kokkos::AUTO()),
          KOKKOS_LAMBDA(
              const Kokkos::TeamPolicy<
                  Kokkos::DefaultExecutionSpace>::member_type &teamMember) {
            const int rank = teamMember.league_rank();
            const int nodeJ = mFarMatJ(workingNode(rank));
            const int indexJStart = mClusterTree(nodeJ, 2);
            const int indexJEnd = mClusterTree(nodeJ, 3);

            const int colSize = indexJEnd - indexJStart;

            const int qkOffset =
                workingNodeQMatOffset(rank, workingNodeIteration(rank));
            const int ik =
                workingNodeSelectedRowIdx(rank, workingNodeIteration(rank));

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(teamMember, colSize * 9),
                [&](const int j) {
                  const int index = qkOffset + j / 9;
                  const int k = j % 9;
                  const int row = k / 3;
                  const int col = k % 3;

                  double sum = 0.0;
                  for (int l = 0; l < innerNumIter - 1; l++) {
                    const int indexL = workingNodeQMatOffset(rank, l) + j / 9;
                    const int cMatOffsetIk =
                        workingNodeCMatOffset(rank, l) + ik;
                    for (int m = 0; m < 3; m++)
                      sum += cMatPool(cMatOffsetIk, 3 * row + m) *
                             qMatPool(indexL, 3 * m + col);
                  }
                  qMatPool(index, k) -= sum;
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

            const int nodeJ = mFarMatJ(workingNode(rank));
            const int indexJStart = mClusterTree(nodeJ, 2);
            const int indexJEnd = mClusterTree(nodeJ, 3);
            const int colSize = indexJEnd - indexJStart;

            const int qkOffset =
                workingNodeQMatOffset(rank, workingNodeIteration(rank));

            Kokkos::MaxLoc<double, int>::value_type result;
            Kokkos::parallel_reduce(
                Kokkos::TeamVectorRange(teamMember, colSize),
                [&](const int j,
                    Kokkos::MaxLoc<double, int>::value_type &update) {
                  double curNorm = 0.0;
                  const int index = qkOffset + j;

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

                  bool exist = false;
                  for (int l = 0; l <= workingNodeIteration(rank); l++)
                    if (workingNodeSelectedColIdx(rank, l) == j)
                      exist = true;

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

      auto end = std::chrono::steady_clock::now();
      qkNormalizationDuration +=
          std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
              .count();
    }

    // stop criterion
    {
      auto start = std::chrono::steady_clock::now();

      Kokkos::parallel_for(
          Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(workNodeSize,
                                                            Kokkos::AUTO()),
          KOKKOS_LAMBDA(
              const Kokkos::TeamPolicy<
                  Kokkos::DefaultExecutionSpace>::member_type &teamMember) {
            const int rank = teamMember.league_rank();

            const int ckOffset =
                workingNodeCMatOffset(rank, workingNodeIteration(rank));
            const int qkOffset =
                workingNodeQMatOffset(rank, workingNodeIteration(rank));

            const int nodeI = mFarMatI(workingNode(rank));
            const int indexIStart = mClusterTree(nodeI, 2);
            const int indexIEnd = mClusterTree(nodeI, 3);
            const int rowSize = indexIEnd - indexIStart;

            const int nodeJ = mFarMatJ(workingNode(rank));
            const int indexJStart = mClusterTree(nodeJ, 2);
            const int indexJEnd = mClusterTree(nodeJ, 3);
            const int colSize = indexJEnd - indexJStart;

            ArrReduce ckArrReduce, qkArrReduce;

            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(teamMember, rowSize * 6),
                [&](const int j, ArrReduce &tSum) {
                  const int row = j / 6;
                  const int k = j % 6;

                  if (k < 3)
                    tSum.values[k] += cMatPool(ckOffset + row, k) *
                                          cMatPool(ckOffset + row, k) +
                                      cMatPool(ckOffset + row, k + 3) *
                                          cMatPool(ckOffset + row, k + 3) +
                                      cMatPool(ckOffset + row, k + 6) *
                                          cMatPool(ckOffset + row, k + 6);
                  else
                    tSum.values[k] +=
                        cMatPool(ckOffset + row, k % 3) *
                            cMatPool(ckOffset + row, (k + 1) % 3) +
                        cMatPool(ckOffset + row, k % 3 + 3) *
                            cMatPool(ckOffset + row, (k + 1) % 3 + 3) +
                        cMatPool(ckOffset + row, k % 3 + 6) *
                            cMatPool(ckOffset + row, (k + 1) % 3 + 6);
                },
                Kokkos::Sum<ArrReduce>(ckArrReduce));

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
            workingNodeIteration(i)++;
            if (nu2(i) < mu2(i) * epsilon2 ||
                workingNodeIteration(i) >= maxIter) {
              stopNode(i) = -1;
            } else {
              stopNode(i) = 0;
            }
          });
      Kokkos::fence();

      {
        int iterationCheckResult = 0;
        Kokkos::parallel_reduce(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workNodeSize),
            KOKKOS_LAMBDA(const int i, int &tIterationCheckResult) {
              if (workingNodeIteration(i) > maxIter)
                tIterationCheckResult++;
            },
            Kokkos::Sum<int>(iterationCheckResult));
        Kokkos::fence();

        if (iterationCheckResult > 0) {
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

      // stage 1
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

            const int nodeI = mFarMatI(dotProductNode(rank));
            const int indexIStart = mClusterTree(nodeI, 2);
            const int indexIEnd = mClusterTree(nodeI, 3);
            const int workSizeI = indexIEnd - indexIStart;

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

                  for (int k = 0; k < 3; k++)
                    sum += cMatPool(cMatOffset + index, row * 3 + k) *
                           middleMatPool(middleMatOffset + 3 * iter + k);

                  Kokkos::atomic_add(&u(indexIStart + index, row), sum);
                });
          });
      Kokkos::fence();

      // stage 2, consider the symmetry property
      if (mUseSymmetry) {
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

        Kokkos::parallel_for(
            Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(dotSize,
                                                              Kokkos::AUTO()),
            KOKKOS_LAMBDA(
                const Kokkos::TeamPolicy<
                    Kokkos::DefaultExecutionSpace>::member_type &teamMember) {
              const int rank = teamMember.league_rank();
              const int workingNodeRank = dotProductRank(rank);

              const int nodeJ = mFarMatJ(dotProductNode(rank));
              const int indexJStart = mClusterTree(nodeJ, 2);
              const int indexJEnd = mClusterTree(nodeJ, 3);
              const int workSizeJ = indexJEnd - indexJStart;

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

                    Kokkos::atomic_add(&u(indexJStart + index, row), sum);
                  });
            });
        Kokkos::fence();
      }

      // post check
      if (postCheck) {
        Kokkos::parallel_for(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, dotSize),
            KOKKOS_LAMBDA(const int rank) { uDotCheck(rank) = 0; });
        Kokkos::fence();

        Kokkos::parallel_for(
            Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(dotSize,
                                                              Kokkos::AUTO()),
            KOKKOS_LAMBDA(
                const Kokkos::TeamPolicy<
                    Kokkos::DefaultExecutionSpace>::member_type &teamMember) {
              const int rank = teamMember.league_rank();

              const int nodeI = mFarMatI(dotProductNode(rank));
              const int indexIStart = mClusterTree(nodeI, 2);
              const int indexIEnd = mClusterTree(nodeI, 3);
              const int workSizeI = indexIEnd - indexIStart;

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

      if (maxInnerNumIter < innerNumIter) {
        maxInnerNumIter = innerNumIter;
      }

      innerNumIter = 0;
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

      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,
                                                             newWorkNodeSize),
          KOKKOS_LAMBDA(const int rank) {
            workingNode(rank) = workingNodeCopy(workingNodeCopyOffset(rank));
          });
      Kokkos::fence();

      // C offset
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

      // Q offset
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
  MPI_Allreduce(MPI_IN_PLACE, &totalNumQuery, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &totalNumIter, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &maxInnerNumIter, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);

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