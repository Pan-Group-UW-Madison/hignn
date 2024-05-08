#include "HignnModel.hpp"

void HignnModel::Build() {
  MPI_Barrier(MPI_COMM_WORLD);
  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();

  if (mMPIRank == 0)
    std::cout << "start of Build" << std::endl;

  Kokkos::deep_copy(*mCoordMirrorPtr, *mCoordPtr);

  mReorderedMap.resize(GetCount());

  int initialLevel = floor(log(mMPISize) / log(2));
  int initialNodeSize = pow(2, initialLevel + 2) - 1;
  int maxWorkSize = pow(2, initialLevel);

  HostIndexMatrix initialClusterTree("initialClusterTree", initialNodeSize, 5);

  // initial sequential stage
  if (mMPIRank == 0) {
    std::queue<std::size_t> nodeList;

    nodeList.emplace(0);

    for (int i = 0; i < initialNodeSize; i++)
      for (int j = 0; j < 5; j++)
        initialClusterTree(i, j) = 0;

#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < GetCount(); i++)
      mReorderedMap[i] = i;

    initialClusterTree(0, 2) = 0;
    initialClusterTree(0, 3) = GetCount();
    initialClusterTree(0, 4) = 0;

    int newNodeNum = 1;

    while (nodeList.size() > 0) {
      int node = nodeList.front();
      nodeList.pop();

      int nodeDividend = 0;

      if ((initialClusterTree(node, 3) - initialClusterTree(node, 2)) >
          mBlockSize) {
        nodeDividend = Divide(initialClusterTree(node, 2),
                              initialClusterTree(node, 3), mReorderedMap, true);
      }

      if (nodeDividend == 0) {
        initialClusterTree(node, 0) = 0;
        initialClusterTree(node, 1) = 0;
      } else {
        initialClusterTree(node, 0) = newNodeNum++;
        initialClusterTree(node, 1) = newNodeNum++;

        initialClusterTree(initialClusterTree(node, 0), 2) =
            initialClusterTree(node, 2);
        initialClusterTree(initialClusterTree(node, 0), 3) = nodeDividend;
        initialClusterTree(initialClusterTree(node, 1), 2) = nodeDividend;
        initialClusterTree(initialClusterTree(node, 1), 3) =
            initialClusterTree(node, 3);

        initialClusterTree(initialClusterTree(node, 0), 4) =
            initialClusterTree(node, 4) + 1;
        initialClusterTree(initialClusterTree(node, 1), 4) =
            initialClusterTree(node, 4) + 1;

        if ((int)initialClusterTree(initialClusterTree(node, 0), 4) <
            initialLevel) {
          nodeList.push(initialClusterTree(node, 0));
          nodeList.push(initialClusterTree(node, 1));
        }
      }
    }
  }

  MPI_Bcast(initialClusterTree.data(), initialNodeSize * 5, MPI_UNSIGNED_LONG,
            0, MPI_COMM_WORLD);
  MPI_Bcast(mReorderedMap.data(), mReorderedMap.size(), MPI_UNSIGNED_LONG, 0,
            MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  HostIndexMatrix mClusterTree("clusterTree", GetCount(), 5);

  int newNodeNum = 1;

  // parallel stage
  if (mMPIRank < maxWorkSize) {
    int startNode = pow(2, initialLevel) - 1 + mMPIRank;
    for (int j = 0; j < 5; j++)
      mClusterTree(0, j) = initialClusterTree(startNode, j);

    std::queue<std::size_t> nodeList;

    nodeList.emplace(0);

    std::vector<std::size_t> nodeDividend(50);
    std::vector<std::size_t> selectedNode(50);

    while (nodeList.size() > 0) {
      int numSelectedNode = std::min((std::size_t)50, nodeList.size());
      for (int i = 0; i < numSelectedNode; i++) {
        selectedNode[i] = nodeList.front();
        nodeList.pop();
      }

      if (numSelectedNode <= 16)
        for (int i = 0; i < numSelectedNode; i++) {
          size_t node = selectedNode[i];

          if ((mClusterTree(node, 3) - mClusterTree(node, 2)) > mBlockSize) {
            nodeDividend[i] =
                Divide(mClusterTree(node, 2), mClusterTree(node, 3),
                       mReorderedMap, true);
          } else {
            nodeDividend[i] = 0;
          }
        }
      else
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < numSelectedNode; i++) {
          size_t node = selectedNode[i];

          if ((mClusterTree(node, 3) - mClusterTree(node, 2)) > mBlockSize) {
            nodeDividend[i] =
                Divide(mClusterTree(node, 2), mClusterTree(node, 3),
                       mReorderedMap, false);
          } else {
            nodeDividend[i] = 0;
          }
        }

      for (int i = 0; i < numSelectedNode; i++) {
        auto node = selectedNode[i];

        if (nodeDividend[i] == 0) {
          mClusterTree(node, 0) = 0;
          mClusterTree(node, 1) = 0;
        } else {
          mClusterTree(node, 0) = newNodeNum++;
          mClusterTree(node, 1) = newNodeNum++;

          mClusterTree(mClusterTree(node, 0), 2) = mClusterTree(node, 2);
          mClusterTree(mClusterTree(node, 0), 3) = nodeDividend[i];
          mClusterTree(mClusterTree(node, 1), 2) = nodeDividend[i];
          mClusterTree(mClusterTree(node, 1), 3) = mClusterTree(node, 3);

          mClusterTree(mClusterTree(node, 0), 4) = mClusterTree(node, 4) + 1;
          mClusterTree(mClusterTree(node, 1), 4) = mClusterTree(node, 4) + 1;

          nodeList.push(mClusterTree(node, 0));
          nodeList.push(mClusterTree(node, 1));
        }
      }
    }
  }

  newNodeNum--;
  mClusterTreeSize = newNodeNum;

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &mClusterTreeSize, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                MPI_COMM_WORLD);

  mClusterTreeSize += initialNodeSize;

  mClusterTreePtr = std::make_shared<DeviceIndexMatrix>(
      DeviceIndexMatrix("mClusterTreePtr", mClusterTreeSize, 5));
  mClusterTreeMirrorPtr = std::make_shared<DeviceIndexMatrix::HostMirror>();
  *mClusterTreeMirrorPtr = Kokkos::create_mirror_view(*mClusterTreePtr);

  auto &hostClusterTree = *mClusterTreeMirrorPtr;

  for (int i = 0; i < initialNodeSize; i++)
    for (int j = 0; j < 5; j++)
      hostClusterTree(i, j) = initialClusterTree(i, j);

  size_t offset = initialNodeSize;
  size_t rankSize = 0;

  for (int rank = 0; rank < maxWorkSize; rank++) {
    if (mMPIRank == rank) {
#pragma omp parallel for schedule(auto)
      for (size_t i = 0; i < (size_t)newNodeNum; i++) {
        if (mClusterTree(i + 1, 0) != 0) {
          hostClusterTree(offset + i, 0) = mClusterTree(i + 1, 0) + offset - 1;
          hostClusterTree(offset + i, 1) = mClusterTree(i + 1, 1) + offset - 1;
        } else {
          hostClusterTree(offset + i, 0) = 0;
          hostClusterTree(offset + i, 1) = 0;
        }
        for (int j = 2; j < 5; j++)
          hostClusterTree(offset + i, j) = mClusterTree(i + 1, j);
      }
      rankSize = newNodeNum;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&rankSize, 1, MPI_UNSIGNED_LONG, rank, MPI_COMM_WORLD);
    // due to the layout property of Kokkos, we need to broadcast the data in
    // multiple times
    for (int i = 0; i < 5; i++)
      MPI_Bcast(hostClusterTree.data() + i * mClusterTreeSize + offset,
                rankSize, MPI_UNSIGNED_LONG, rank, MPI_COMM_WORLD);

    int reorderedNode = pow(2, initialLevel) - 1 + rank;
    size_t reorderedMapSize = initialClusterTree(reorderedNode, 3) -
                              initialClusterTree(reorderedNode, 2);
    MPI_Bcast(mReorderedMap.data() + initialClusterTree(reorderedNode, 2),
              reorderedMapSize, MPI_UNSIGNED_LONG, rank, MPI_COMM_WORLD);

    hostClusterTree(reorderedNode, 0) = offset;
    hostClusterTree(reorderedNode, 1) = offset + 1;

    if (mMPIRank == rank)
      offset += newNodeNum;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&offset, 1, MPI_UNSIGNED_LONG, rank, MPI_COMM_WORLD);
  }

  Kokkos::deep_copy(*mClusterTreePtr, hostClusterTree);

  Reorder(mReorderedMap);

  MPI_Barrier(MPI_COMM_WORLD);
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

  if (mMPIRank == 0) {
    std::cout << "Build time: " << (double)duration / 1e6 << "s" << std::endl;

    std::cout << "end of Build" << std::endl;
  }
}