#include "HignnModel.hpp"

void HignnModel::CloseFarCheck() {
  MPI_Barrier(MPI_COMM_WORLD);
  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();

  if (mMPIRank == 0)
    std::cout << "start of CloseFarCheck" << std::endl;

  mLeafNodeList.clear();

  auto &mClusterTree = *mClusterTreeMirrorPtr;

  HostFloatMatrix mAux;
  Kokkos::resize(mAux, mClusterTree.extent(0), 6);

  // init mAux
  for (size_t i = 0; i < mAux.extent(0); i++)
    mAux(i, 0) = -std::numeric_limits<float>::max();

  int initialLevel = floor(log(mMPISize) / log(2));
  int initialNodeSize = pow(2, initialLevel + 1) - 1;
  int maxWorkSize = pow(2, initialLevel);

  // go over all nodes, calculate aux based on the tree structure.
  // parallel stage
  if (mMPIRank < maxWorkSize) {
    std::stack<std::size_t> workStack;
    std::vector<std::size_t> computeAuxStack;

    int startNode = pow(2, initialLevel) - 1 + mMPIRank;
    workStack.push(startNode);

    while (workStack.size() != 0) {
      auto node = workStack.top();

      workStack.pop();

      if (mClusterTree(node, 0) != 0) {
        workStack.push(mClusterTree(node, 0));
        workStack.push(mClusterTree(node, 1));
      } else {
        computeAuxStack.push_back(node);
      }
    }

#pragma omp parallel for schedule(dynamic, 10)
    for (size_t i = 0; i < computeAuxStack.size(); i++) {
      auto node = computeAuxStack[i];

      std::vector<float> aux(6);
      ComputeAux(mClusterTree(node, 2), mClusterTree(node, 3), aux);

      mAux(node, 0) = aux[0];
      mAux(node, 1) = aux[1];
      mAux(node, 2) = aux[2];
      mAux(node, 3) = aux[3];
      mAux(node, 4) = aux[4];
      mAux(node, 5) = aux[5];
    }

    workStack.push(startNode);
    while (workStack.size() != 0) {
      auto node = workStack.top();
      if (mClusterTree(node, 0) != 0) {
        // check if child nodes have calculated aux.
        bool canContinue = false;
        if (mAux(mClusterTree(node, 0), 0) ==
            -std::numeric_limits<float>::max()) {
          workStack.push(mClusterTree(node, 0));

          canContinue = true;
        }

        if (mAux(mClusterTree(node, 1), 0) ==
            -std::numeric_limits<float>::max()) {
          workStack.push(mClusterTree(node, 1));

          canContinue = true;
        }

        if (!canContinue) {
          mAux(node, 0) = std::min(mAux(mClusterTree(node, 0), 0),
                                   mAux(mClusterTree(node, 1), 0));
          mAux(node, 1) = std::max(mAux(mClusterTree(node, 0), 1),
                                   mAux(mClusterTree(node, 1), 1));
          mAux(node, 2) = std::min(mAux(mClusterTree(node, 0), 2),
                                   mAux(mClusterTree(node, 1), 2));
          mAux(node, 3) = std::max(mAux(mClusterTree(node, 0), 3),
                                   mAux(mClusterTree(node, 1), 3));
          mAux(node, 4) = std::min(mAux(mClusterTree(node, 0), 4),
                                   mAux(mClusterTree(node, 1), 4));
          mAux(node, 5) = std::max(mAux(mClusterTree(node, 0), 5),
                                   mAux(mClusterTree(node, 1), 5));

          workStack.pop();
        }
      } else
        workStack.pop();
    }
  }

  for (int rank = 0; rank < maxWorkSize; rank++) {
    int reorderedNode = pow(2, initialLevel) - 1 + rank;
    const size_t nodeStart = mClusterTree(reorderedNode, 0);
    const size_t nodeEnd = (rank == maxWorkSize - 1)
                               ? mClusterTree.extent(0)
                               : mClusterTree(reorderedNode + 1, 0);

    MPI_Bcast(mAux.data() + 6 * nodeStart, 6 * (nodeEnd - nodeStart), MPI_FLOAT,
              rank, MPI_COMM_WORLD);
  }

  // sequential stage
  {
    std::stack<std::size_t> workStack;
    workStack.push(0);
    while (workStack.size() != 0) {
      auto node = workStack.top();
      if (mClusterTree(node, 0) != 0) {
        // check if child nodes have calculated aux.
        bool canContinue = false;
        if (mAux(mClusterTree(node, 0), 0) ==
            -std::numeric_limits<float>::max()) {
          workStack.push(mClusterTree(node, 0));

          canContinue = true;
        }

        if (mAux(mClusterTree(node, 1), 0) ==
            -std::numeric_limits<float>::max()) {
          workStack.push(mClusterTree(node, 1));

          canContinue = true;
        }

        if (!canContinue) {
          mAux(node, 0) = std::min(mAux(mClusterTree(node, 0), 0),
                                   mAux(mClusterTree(node, 1), 0));
          mAux(node, 1) = std::max(mAux(mClusterTree(node, 0), 1),
                                   mAux(mClusterTree(node, 1), 1));
          mAux(node, 2) = std::min(mAux(mClusterTree(node, 0), 2),
                                   mAux(mClusterTree(node, 1), 2));
          mAux(node, 3) = std::max(mAux(mClusterTree(node, 0), 3),
                                   mAux(mClusterTree(node, 1), 3));
          mAux(node, 4) = std::min(mAux(mClusterTree(node, 0), 4),
                                   mAux(mClusterTree(node, 1), 4));
          mAux(node, 5) = std::max(mAux(mClusterTree(node, 0), 5),
                                   mAux(mClusterTree(node, 1), 5));

          workStack.pop();
        }
      } else {
        std::vector<float> aux(6);
        ComputeAux(mClusterTree(node, 2), mClusterTree(node, 3), aux);

        mAux(node, 0) = aux[0];
        mAux(node, 1) = aux[1];
        mAux(node, 2) = aux[2];
        mAux(node, 3) = aux[3];
        mAux(node, 4) = aux[4];
        mAux(node, 5) = aux[5];

        workStack.pop();
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // estimate far and close pair size based on aux.
  // close far check
  std::vector<std::vector<int>> farMat(mClusterTree.extent(0));
  std::vector<std::vector<int>> closeMat(mClusterTree.extent(0));

  std::size_t total_entry = 0;

  closeMat[0].push_back(0);
  std::queue<std::size_t> nodeList;
  nodeList.emplace(0);
  while (nodeList.size() != 0) {
    auto node = nodeList.front();
    auto i = node;
    nodeList.pop();

    if (mClusterTree(node, 0) != 0) {
      std::vector<int> childCloseMat;

      for (size_t j = 0; j < closeMat[i].size(); j++) {
        bool isFar = CloseFarCheck(mAux, node, closeMat[i][j]);

        std::size_t nodeSizeI = mClusterTree(node, 3) - mClusterTree(node, 2);
        std::size_t nodeSizeJ =
            mClusterTree(closeMat[i][j], 3) - mClusterTree(closeMat[i][j], 2);

        isFar = isFar && nodeSizeI < mMaxRelativeCoord &&
                nodeSizeJ < mMaxRelativeCoord;

        if (isFar) {
          farMat[i].push_back(closeMat[i][j]);

          total_entry += nodeSizeI * nodeSizeJ;
        } else {
          if (mClusterTree(closeMat[i][j], 0) != 0) {
            childCloseMat.push_back(mClusterTree(closeMat[i][j], 0));
            childCloseMat.push_back(mClusterTree(closeMat[i][j], 1));
          } else {
            childCloseMat.push_back(closeMat[i][j]);
          }
        }
      }
      closeMat[mClusterTree(node, 0)] = childCloseMat;
      closeMat[mClusterTree(node, 1)] = childCloseMat;

      nodeList.emplace(mClusterTree(node, 0));
      nodeList.emplace(mClusterTree(node, 1));
    } else {
      std::vector<int> newCloseMat;

      for (size_t j = 0; j < closeMat[i].size(); j++) {
        bool isFar = CloseFarCheck(mAux, node, closeMat[i][j]);

        std::size_t nodeSizeI = mClusterTree(node, 3) - mClusterTree(node, 2);
        std::size_t nodeSizeJ =
            mClusterTree(closeMat[i][j], 3) - mClusterTree(closeMat[i][j], 2);

        total_entry += nodeSizeI * nodeSizeJ;

        if (isFar) {
          // need to make sure the column node is small enough
          if (nodeSizeJ < mMaxRelativeCoord)
            farMat[i].push_back(closeMat[i][j]);
          else {
            std::stack<int> workChildStack;
            workChildStack.push(closeMat[i][j]);

            while (workChildStack.size() != 0) {
              auto childNode = workChildStack.top();
              workChildStack.pop();
              size_t nodeSize =
                  mClusterTree(childNode, 3) - mClusterTree(childNode, 2);

              if (nodeSize < mMaxRelativeCoord) {
                farMat[i].push_back(childNode);
              } else {
                workChildStack.push(mClusterTree(childNode, 0));
                workChildStack.push(mClusterTree(childNode, 1));
              }
            }
          }
        } else {
          // need to make sure this is a leaf node for close mat.
          if (mClusterTree(closeMat[i][j], 0) == 0)
            newCloseMat.push_back(closeMat[i][j]);
          else {
            std::vector<int> childCloseMat;
            std::stack<int> workChildStack;
            workChildStack.push(closeMat[i][j]);

            while (workChildStack.size() != 0) {
              auto childNode = workChildStack.top();
              workChildStack.pop();

              if (mClusterTree(childNode, 0) != 0) {
                workChildStack.push(mClusterTree(childNode, 0));
                workChildStack.push(mClusterTree(childNode, 1));
              } else {
                childCloseMat.push_back(childNode);
              }
            }

            for (size_t k = 0; k < childCloseMat.size(); k++)
              newCloseMat.push_back(childCloseMat[k]);
          }
        }
      }

      closeMat[i] = newCloseMat;

      mLeafNodeList.push_back(node);
    }
  }

  if (mMPIRank == 0) {
    std::cout << "total entry: " << total_entry << std::endl;

    std::cout << "num of leaf nodes: " << mLeafNodeList.size() << std::endl;
  }

  // split leaf node among mpi ranks
  size_t totalCloseEntry = 0;
  size_t totalClosePair = 0;
  std::vector<size_t> leafNode;
  std::vector<size_t> closeMatI;
  std::vector<size_t> closeMatJ;
  closeMatI.push_back(0);
  mMaxCloseDotBlockSize = 0;
  for (size_t i = 0; i < mLeafNodeList.size(); i++) {
    int nodeI = mLeafNodeList[i];
    int nodeSizeI = mClusterTree(nodeI, 3) - mClusterTree(nodeI, 2);
    int colSize = closeMat[mLeafNodeList[i]].size();
    int leafNodeCounter = 0;
    for (int j = 0; j < colSize; j++) {
      int nodeJ = closeMat[nodeI][j];
      // consider the symmetry property
      if (mUseSymmetry) {
        if (nodeJ >= nodeI) {
          if (totalClosePair % (size_t)mMPISize == (size_t)mMPIRank) {
            int nodeSizeJ = mClusterTree(nodeJ, 3) - mClusterTree(nodeJ, 2);

            leafNodeCounter++;

            closeMatJ.push_back(nodeJ);

            if (nodeI == nodeJ)
              totalCloseEntry += nodeSizeI * nodeSizeJ;
            else
              totalCloseEntry += 2 * nodeSizeI * nodeSizeJ;

            if (nodeSizeI * nodeSizeJ > mMaxCloseDotBlockSize)
              mMaxCloseDotBlockSize = nodeSizeI * nodeSizeJ;
          }

          totalClosePair++;
        }
      } else {
        if (totalClosePair % (size_t)mMPISize == (size_t)mMPIRank) {
          int nodeSizeJ = mClusterTree(nodeJ, 3) - mClusterTree(nodeJ, 2);

          leafNodeCounter++;

          closeMatJ.push_back(nodeJ);

          if (nodeI == nodeJ)
            totalCloseEntry += nodeSizeI * nodeSizeJ;
          else
            totalCloseEntry += 2 * nodeSizeI * nodeSizeJ;

          if (nodeSizeI * nodeSizeJ > mMaxCloseDotBlockSize)
            mMaxCloseDotBlockSize = nodeSizeI * nodeSizeJ;
        }

        totalClosePair++;
      }
    }

    if (leafNodeCounter != 0) {
      leafNode.push_back(nodeI);
      closeMatI.push_back(closeMatI.back() + leafNodeCounter);
    }
  }

  mLeafNodePtr =
      std::make_shared<DeviceIndexVector>("mLeafNode", leafNode.size());
  auto &mLeafNode = *mLeafNodePtr;
  DeviceIndexVector::HostMirror hostLeafNode =
      Kokkos::create_mirror_view(*mLeafNodePtr);

  for (size_t i = 0; i < leafNode.size(); i++) {
    hostLeafNode(i) = leafNode[i];
  }

  mCloseMatIPtr =
      std::make_shared<DeviceIndexVector>("mCloseMatI", closeMatI.size());
  auto &mCloseMatI = *mCloseMatIPtr;
  DeviceIndexVector::HostMirror hostCloseMatI =
      Kokkos::create_mirror_view(*mCloseMatIPtr);

  for (size_t i = 0; i < closeMatI.size(); i++) {
    hostCloseMatI(i) = closeMatI[i];
  }

  mCloseMatJPtr =
      std::make_shared<DeviceIndexVector>("mCloseMatJ", closeMatJ.size());
  auto &mCloseMatJ = *mCloseMatJPtr;
  DeviceIndexVector::HostMirror hostCloseMatJ =
      Kokkos::create_mirror_view(*mCloseMatJPtr);

  for (size_t i = 0; i < closeMatJ.size(); i++) {
    hostCloseMatJ(i) = closeMatJ[i];
  }

  MPI_Allreduce(MPI_IN_PLACE, &totalCloseEntry, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                MPI_COMM_WORLD);

  if (mMPIRank == 0)
    std::cout << "Total close pair: " << totalClosePair << std::endl;

  if (mMPIRank == 0)
    std::cout << "Total close entry: " << totalCloseEntry << std::endl;

  Kokkos::deep_copy(mCloseMatI, hostCloseMatI);
  Kokkos::deep_copy(mCloseMatJ, hostCloseMatJ);
  Kokkos::deep_copy(mLeafNode, hostLeafNode);

  std::size_t totalFarSize = 0;
  std::size_t totalFarNode = 0;
  for (size_t i = 0; i < farMat.size(); i++) {
    if (farMat[i].size() != 0) {
      totalFarSize += farMat[i].size();
      totalFarNode++;
    }
  }
  if (mMPIRank == 0) {
    std::cout << "Total far node: " << totalFarNode << std::endl;
    std::cout << "Total far pair: " << totalFarSize << std::endl;
  }

  // split far pair among mpi ranks
  std::vector<size_t> farMatI;
  std::vector<size_t> farMatJ;

  int counter = 0;
  for (size_t i = 0; i < farMat.size(); i++) {
    int nodeI = i;
    for (size_t j = 0; j < farMat[i].size(); j++) {
      int nodeJ = farMat[i][j];

      // consider the symmetry property
      if (mUseSymmetry) {
        if (nodeJ >= nodeI) {
          if (counter % (std::size_t)mMPISize == (std::size_t)mMPIRank) {
            farMatI.push_back(nodeI);
            farMatJ.push_back(nodeJ);
          }
          counter++;
        }
      } else {
        if (counter % (std::size_t)mMPISize == (std::size_t)mMPIRank) {
          farMatI.push_back(nodeI);
          farMatJ.push_back(nodeJ);
        }
        counter++;
      }
    }
  }

  mFarMatIPtr = std::make_shared<DeviceIndexVector>("mFarMatI", farMatI.size());
  auto &mFarMatI = *mFarMatIPtr;

  mFarMatJPtr = std::make_shared<DeviceIndexVector>("mFarMatJ", farMatJ.size());
  auto &mFarMatJ = *mFarMatJPtr;

  DeviceIndexVector::HostMirror farMatIMirror =
      Kokkos::create_mirror_view(mFarMatI);
  DeviceIndexVector::HostMirror farMatJMirror =
      Kokkos::create_mirror_view(mFarMatJ);

  for (size_t i = 0; i < farMatI.size(); i++) {
    farMatIMirror(i) = farMatI[i];
    farMatJMirror(i) = farMatJ[i];
  }

  std::size_t farDotQueryNum = 0;
  std::size_t maxSingleNodeSize = 0;
  for (size_t i = 0; i < farMatIMirror.extent(0); i++) {
    std::size_t nodeISize =
        mClusterTree(farMatIMirror(i), 3) - mClusterTree(farMatIMirror(i), 2);
    std::size_t nodeJSize =
        mClusterTree(farMatJMirror(i), 3) - mClusterTree(farMatJMirror(i), 2);
    farDotQueryNum += 2 * nodeISize * nodeJSize;

    if (nodeISize > maxSingleNodeSize)
      maxSingleNodeSize = nodeISize;
    if (nodeJSize > maxSingleNodeSize)
      maxSingleNodeSize = nodeJSize;
  }
  MPI_Allreduce(MPI_IN_PLACE, &farDotQueryNum, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  if (mMPIRank == 0)
    std::cout << "Total far entry: " << farDotQueryNum << std::endl;

  MPI_Allreduce(MPI_IN_PLACE, &maxSingleNodeSize, 1, MPI_UNSIGNED_LONG, MPI_MAX,
                MPI_COMM_WORLD);
  if (mMPIRank == 0)
    std::cout << "Max single node size: " << maxSingleNodeSize << std::endl;

  Kokkos::deep_copy(mFarMatI, farMatIMirror);
  Kokkos::deep_copy(mFarMatJ, farMatJMirror);

  MPI_Barrier(MPI_COMM_WORLD);
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  if (mMPIRank == 0) {
    std::cout << "Time for building close and far matrix: "
              << (double)duration / 1e6 << "s" << std::endl;
  }
}