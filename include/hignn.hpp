#ifndef HIGNN_HPP
#define HIGNN_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <execution>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <queue>
#include <stack>
#include <vector>

#include <torch/script.h>

using namespace std::chrono;

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>

#include "Typedef.hpp"

class Problem {
protected:
  std::shared_ptr<DeviceFloatMatrix> mCoordPtr;
  std::shared_ptr<DeviceFloatMatrix::HostMirror> mCoordMirrorPtr;

  std::shared_ptr<DeviceIndexMatrix> mClusterTreePtr;
  std::shared_ptr<DeviceIndexMatrix::HostMirror> mClusterTreeMirrorPtr;

  std::shared_ptr<DeviceIndexVector> mCloseMatIPtr;
  std::shared_ptr<DeviceIndexVector> mCloseMatJPtr;

  std::shared_ptr<DeviceIndexVector> mFarMatIPtr;
  std::shared_ptr<DeviceIndexVector> mFarMatJPtr;

  std::shared_ptr<DeviceIndexVector> mLeafNodePtr;

  std::vector<std::size_t> mReorderedMap;

  int mBlockSize;
  int mDim;

  std::size_t clusterTreeSize;
  torch::jit::script::Module module;

protected:
  std::size_t GetCount() { return mCoordPtr->extent(0); }

  void ComputeAux(const std::size_t first, const std::size_t last,
                  std::vector<float> &aux) {
    auto &mVertexMirror = *mCoordMirrorPtr;

    for (int d = 0; d < mDim; d++) {
      aux[2 * d] = std::numeric_limits<float>::max();
      aux[2 * d + 1] = -std::numeric_limits<float>::max();
    }

    for (int i = first; i < last; i++)
      for (int d = 0; d < mDim; d++) {
        aux[2 * d] = (aux[2 * d] > mVertexMirror(i, d)) ? mVertexMirror(i, d)
                                                        : aux[2 * d];
        aux[2 * d + 1] = (aux[2 * d + 1] < mVertexMirror(i, d))
                             ? mVertexMirror(i, d)
                             : aux[2 * d + 1];
      }
  }

  std::size_t Divide(const std::size_t first, const std::size_t last,
                     const int axis, std::vector<std::size_t> &reorderedMap) {
    auto &mVertexMirror = *mCoordMirrorPtr;

    const std::size_t L = last - first;

    // double mean = 0.0;
    // std::vector<double> temp(L);

    // for (auto i = first; i < last; i++)
    //   mean += mVertexMirror(reorderedMap[i], axis);

    // mean /= (double)L;

    // for (auto i = 0; i < L; i++)
    //   temp[i] = mVertexMirror(reorderedMap[i + first], axis) - mean;

    std::vector<float> mean(mDim, 0.0);
    std::vector<float> temp(L);

    for (auto i = first; i < last; i++)
      for (int d = 0; d < mDim; d++)
        mean[d] += mVertexMirror(reorderedMap[i], d);
    for (int d = 0; d < mDim; d++)
      mean[d] /= (float)L;

    Eigen::MatrixXf vertexHat(mDim, L);
    for (int d = 0; d < mDim; d++) {
      for (int i = 0; i < L; i++) {
        vertexHat(d, i) = mVertexMirror(reorderedMap[i + first], d) - mean[d];
      }
    }

    Eigen::JacobiSVD<Eigen::MatrixXf> svd_holder(
        vertexHat, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXf D = svd_holder.singularValues();
    Eigen::MatrixXf U = svd_holder.matrixU();
    Eigen::MatrixXf V = svd_holder.matrixV();

    for (int i = 0; i < L; i++) {
      temp[i] = 0.0;
      for (int d = 0; d < mDim; d++) {
        temp[i] += U(d, 0) * vertexHat(d, i);
      }
    }

    std::vector<std::size_t> newIndex;
    newIndex.resize(temp.size());
    iota(newIndex.begin(), newIndex.end(), 0);
    sort(newIndex.begin(), newIndex.end(),
         [&temp](int i1, int i2) { return temp[i1] < temp[i2]; });
    std::sort(temp.begin(), temp.end());

    // Currently, the reordering is a very stupid implementation. Need to
    // improve it.

    std::vector<std::size_t> copyIndex(L);
    for (auto i = 0; i < L; i++)
      copyIndex[i] = reorderedMap[i + first];

    for (auto i = 0; i < L; i++)
      reorderedMap[i + first] = copyIndex[newIndex[i]];

    auto result = std::upper_bound(temp.begin(), temp.end(), 0);

    return (std::size_t)(result - temp.begin()) + first;
  }

  void Reorder(const std::vector<std::size_t> &reorderedMap) {
    auto &mVertexMirror = *mCoordMirrorPtr;
    auto &mVertex = *mCoordPtr;

    HostFloatMatrix copyVertex;
    Kokkos::resize(copyVertex, reorderedMap.size(), 3);
    for (auto i = 0; i < reorderedMap.size(); i++) {
      copyVertex(i, 0) = mVertexMirror(i, 0);
      copyVertex(i, 1) = mVertexMirror(i, 1);
      copyVertex(i, 2) = mVertexMirror(i, 2);
    }

    for (auto i = 0; i < reorderedMap.size(); i++) {
      mVertexMirror(i, 0) = copyVertex(reorderedMap[i], 0);
      mVertexMirror(i, 1) = copyVertex(reorderedMap[i], 1);
      mVertexMirror(i, 2) = copyVertex(reorderedMap[i], 2);
    }

    Kokkos::deep_copy(mVertex, mVertexMirror);
  }

public:
  Problem(DeviceFloatMatrix coord, const int blockSize) {
    mCoordPtr = std::make_shared<DeviceFloatMatrix>(coord);
    mCoordMirrorPtr = std::make_shared<DeviceFloatMatrix::HostMirror>();
    *mCoordMirrorPtr = Kokkos::create_mirror_view(*mCoordPtr);

    mBlockSize = blockSize;
    mDim = 3;

    // load script module
    module = torch::jit::load("3D_force_UB_max600_try2.pt");
    module.to(at::kCUDA);

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(torch::kCUDA, 0)
                       .requires_grad(false);
    torch::Tensor testTensor = torch::ones({500000, 3}, options);
    std::vector<c10::IValue> inputs;
    inputs.push_back(testTensor);

    auto testResult = module.forward(inputs);
  }

  void Update() {
    Build();

    CloseFarCheck();
  }

  void Build() {
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();

    std::cout << "start of Build" << std::endl;

    Kokkos::deep_copy(*mCoordMirrorPtr, *mCoordPtr);

    std::queue<std::size_t> nodeList;

    nodeList.emplace(0);

    std::size_t estimatedSize = GetCount();

    HostIndexMatrix mClusterTree;
    Kokkos::resize(mClusterTree, estimatedSize, 5);

    for (std::size_t i = 0; i < mClusterTree.extent(0); i++)
      for (int j = 0; j < mClusterTree.extent(1); j++)
        mClusterTree(i, j) = 0;

    mReorderedMap.resize(GetCount());
    for (std::size_t i = 0; i < GetCount(); i++)
      mReorderedMap[i] = i;

    mClusterTree(0, 2) = 0;
    mClusterTree(0, 3) = GetCount();
    mClusterTree(0, 4) = 0;

    int newNodeNum = 1;

    std::vector<std::size_t> nodeDividend(10);
    std::vector<std::size_t> selectedNode(10);

    while (nodeList.size() > 0) {
      int numSelectedNode = std::min((std::size_t)10, nodeList.size());
      for (int i = 0; i < numSelectedNode; i++) {
        selectedNode[i] = nodeList.front();
        nodeList.pop();
      }

#pragma omp parallel for schedule(dynamic)
      for (int i = 0; i < numSelectedNode; i++) {
        auto node = selectedNode[i];

        if ((mClusterTree(node, 3) - mClusterTree(node, 2)) > mBlockSize) {
          nodeDividend[i] = Divide(mClusterTree(node, 2), mClusterTree(node, 3),
                                   mClusterTree(node, 4) % 3, mReorderedMap);
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

    clusterTreeSize = newNodeNum;

    mClusterTreePtr = std::make_shared<DeviceIndexMatrix>();
    Kokkos::resize(*mClusterTreePtr, clusterTreeSize, 5);
    mClusterTreeMirrorPtr = std::make_shared<DeviceIndexMatrix::HostMirror>();
    *mClusterTreeMirrorPtr = Kokkos::create_mirror_view(*mClusterTreePtr);

    auto &hostClusterTree = *mClusterTreeMirrorPtr;

    for (std::size_t i = 0; i < clusterTreeSize; i++)
      for (int j = 0; j < mClusterTree.extent(1); j++)
        hostClusterTree(i, j) = mClusterTree(i, j);

    Kokkos::deep_copy(*mClusterTreePtr, hostClusterTree);

    Reorder(mReorderedMap);

    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    std::cout << "Build time: " << (double)duration / 1e6 << "s" << std::endl;

    std::cout << "end of Build" << std::endl;
  }

  inline bool CloseFarCheck(HostFloatMatrix aux, const std::size_t node1,
                            const std::size_t node2) {
    float diam0 = 0.0;
    float diam1 = 0.0;
    float dist = 0.0;
    float tmp = 0.0;

    bool isFar = false;
    // AABB bounding box intersection check
    if (aux(node1, 0) > aux(node2, 1) || aux(node1, 1) < aux(node2, 0) ||
        aux(node1, 2) > aux(node2, 3) || aux(node1, 3) < aux(node2, 2) ||
        aux(node1, 4) > aux(node2, 5) || aux(node1, 5) < aux(node2, 4))
      isFar = true;
    else
      return false;

    for (int j = 0; j < 3; j++) {
      tmp = aux(node1, 2 * j) - aux(node1, 2 * j + 1);
      diam0 += tmp * tmp;
      tmp = aux(node2, 2 * j) - aux(node2, 2 * j + 1);
      diam1 += tmp * tmp;
      tmp = aux(node1, 2 * j) + aux(node1, 2 * j + 1) - aux(node2, 2 * j) -
            aux(node2, 2 * j + 1);
      dist += tmp * tmp;
    }

    dist *= 0.25;
    if ((dist > diam0) && (dist > diam1)) {
      isFar = true;
    } else {
      isFar = false;
    }

    return isFar;
  }

  void CloseFarCheck() {
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();

    std::cout << "start of CloseFarCheck" << std::endl;

    auto &mClusterTree = *mClusterTreeMirrorPtr;

    HostFloatMatrix mAux;
    Kokkos::resize(mAux, mClusterTree.extent(0), 6);

    // init mAux
    for (int i = 0; i < mAux.extent(0); i++)
      mAux(i, 0) = -std::numeric_limits<float>::max();

    // go over all nodes, calculate aux based on the tree structure.
    std::stack<std::size_t> workStack;
    workStack.push(0);
    while (workStack.size() != 0) {
      auto node = workStack.top();
      if (mClusterTree(node, 0) != 0) {
        // check if child nodes have calculated aux.
        if (mAux(mClusterTree(node, 0), 0) ==
            -std::numeric_limits<float>::max()) {
          workStack.push(mClusterTree(node, 0));
          workStack.push(mClusterTree(node, 1));
          continue;
        } else {
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

    // estimate far and close pair size based on aux.
    std::vector<std::size_t> leafNode;

    // close far check
    std::vector<std::vector<int>> farMat(mClusterTree.extent(0));
    std::vector<std::vector<int>> closeMat(mClusterTree.extent(0));

    std::size_t total_entry = 0;

    closeMat[0].push_back(0);
    for (int i = 0; i < mClusterTree.extent(0); i++) {
      auto node = i;

      if (mClusterTree(node, 0) != 0) {
        std::vector<int> childCloseMat;

        for (int j = 0; j < closeMat[i].size(); j++) {
          bool isFar = CloseFarCheck(mAux, node, closeMat[i][j]);

          if (isFar) {
            farMat[i].push_back(closeMat[i][j]);

            int nodeSizeI = mClusterTree(node, 3) - mClusterTree(node, 2);
            int nodeSizeJ = mClusterTree(closeMat[i][j], 3) -
                            mClusterTree(closeMat[i][j], 2);

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
      } else {
        std::vector<int> newCloseMat;

        for (int j = 0; j < closeMat[i].size(); j++) {
          bool isFar = CloseFarCheck(mAux, node, closeMat[i][j]);

          int nodeSizeI = mClusterTree(node, 3) - mClusterTree(node, 2);
          int nodeSizeJ =
              mClusterTree(closeMat[i][j], 3) - mClusterTree(closeMat[i][j], 2);

          total_entry += nodeSizeI * nodeSizeJ;

          if (isFar)
            farMat[i].push_back(closeMat[i][j]);
          else {
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

              for (int k = 0; k < childCloseMat.size(); k++)
                newCloseMat.push_back(childCloseMat[k]);
            }
          }
        }

        closeMat[i] = newCloseMat;

        leafNode.push_back(node);
      }
    }

    std::cout << "total entry: " << total_entry << std::endl;

    std::cout << "num of leaf nodes: " << leafNode.size() << std::endl;

    mLeafNodePtr = std::make_shared<DeviceIndexVector>();
    Kokkos::resize(*mLeafNodePtr, leafNode.size());

    DeviceIndexVector::HostMirror hostLeafNode =
        Kokkos::create_mirror_view(*mLeafNodePtr);

    auto &mLeafNode = *mLeafNodePtr;

    for (int i = 0; i < leafNode.size(); i++)
      hostLeafNode(i) = leafNode[i];
    Kokkos::deep_copy(mLeafNode, hostLeafNode);

    mCloseMatIPtr = std::make_shared<DeviceIndexVector>();
    auto &mCloseMatI = *mCloseMatIPtr;
    Kokkos::resize(*mCloseMatIPtr, leafNode.size() + 1);

    DeviceIndexVector::HostMirror hostCloseMatI =
        Kokkos::create_mirror_view(*mCloseMatIPtr);
    hostCloseMatI(0) = 0;
    for (int i = 0; i < leafNode.size(); i++)
      hostCloseMatI(i + 1) = hostCloseMatI(i) + closeMat[leafNode[i]].size();

    mCloseMatJPtr = std::make_shared<DeviceIndexVector>();
    auto &mCloseMatJ = *mCloseMatJPtr;
    Kokkos::resize(*mCloseMatJPtr, hostCloseMatI(leafNode.size()));

    DeviceIndexVector::HostMirror hostCloseMatJ =
        Kokkos::create_mirror_view(*mCloseMatJPtr);
    for (int i = 0; i < leafNode.size(); i++) {
      for (int j = 0; j < closeMat[leafNode[i]].size(); j++) {
        hostCloseMatJ(hostCloseMatI(i) + j) = closeMat[leafNode[i]][j];
      }
    }

    std::cout << "Total close pair: " << hostCloseMatI(leafNode.size())
              << std::endl;

    std::size_t totalCloseEntry = 0;
    for (int i = 0; i < leafNode.size(); i++) {
      int nodeI = leafNode[i];
      int nodeSizeI = mClusterTree(nodeI, 3) - mClusterTree(nodeI, 2);
      for (int j = 0; j < closeMat[nodeI].size(); j++) {
        int nodeJ = closeMat[nodeI][j];
        int nodeSizeJ = mClusterTree(nodeJ, 3) - mClusterTree(nodeJ, 2);

        totalCloseEntry += nodeSizeI * nodeSizeJ;
      }
    }

    std::cout << "Total close entry: " << totalCloseEntry << std::endl;

    Kokkos::deep_copy(mCloseMatI, hostCloseMatI);
    Kokkos::deep_copy(mCloseMatJ, hostCloseMatJ);

    std::size_t totalFarSize = 0;
    std::size_t totalFarNode = 0;
    for (int i = 0; i < farMat.size(); i++) {
      if (farMat[i].size() != 0) {
        totalFarSize += farMat[i].size();
        totalFarNode++;
      }
    }
    std::cout << "Total far node: " << totalFarNode << std::endl;
    std::cout << "Total far pair: " << totalFarSize << std::endl;

    mFarMatIPtr = std::make_shared<DeviceIndexVector>();
    auto &mFarMatI = *mFarMatIPtr;
    Kokkos::resize(*mFarMatIPtr, totalFarSize);

    mFarMatJPtr = std::make_shared<DeviceIndexVector>();
    auto &mFarMatJ = *mFarMatJPtr;
    Kokkos::resize(*mFarMatJPtr, totalFarSize);

    DeviceIndexVector::HostMirror farMatIMirror =
        Kokkos::create_mirror_view(mFarMatI);
    DeviceIndexVector::HostMirror farMatJMirror =
        Kokkos::create_mirror_view(mFarMatJ);

    int counter = 0;
    for (int i = 0; i < farMat.size(); i++) {
      for (int j = 0; j < farMat[i].size(); j++) {
        farMatIMirror(counter) = i;
        farMatJMirror(counter) = farMat[i][j];
        counter++;
      }
    }

    std::size_t farDotQueryNum = 0;
    for (int i = 0; i < farMatIMirror.extent(0); i++) {
      std::size_t nodeISize =
          mClusterTree(farMatIMirror(i), 3) - mClusterTree(farMatIMirror(i), 2);
      std::size_t nodeJSize =
          mClusterTree(farMatJMirror(i), 3) - mClusterTree(farMatJMirror(i), 2);
      farDotQueryNum += nodeISize * nodeJSize;
    }
    std::cout << "Total far entry: " << farDotQueryNum << std::endl;

    Kokkos::deep_copy(mFarMatI, farMatIMirror);
    Kokkos::deep_copy(mFarMatJ, farMatJMirror);

    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Time for building close and far matrix: "
              << (double)duration / 1e6 << "s" << std::endl;

    std::cout << "end of CloseDot" << std::endl;
  }

  void CloseDot(DeviceFloatMatrix u, DeviceFloatMatrix f) {
    std::cout << "start of CloseDot" << std::endl;

    double queryDuration = 0;
    double dotDuration = 0;

    const int leafNodeSize = mLeafNodePtr->extent(0);
    const int maxWorkSize = std::min(100, leafNodeSize);
    int workSize = maxWorkSize;

    std::size_t totalNumQuery = 0;
    std::size_t totalNumIter = 0;

    DeviceFloatMatrix relativeCoordPool(
        "relativeCoordPool", maxWorkSize * mBlockSize * mBlockSize, 3);
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
              const Kokkos::TeamPolicy<
                  Kokkos::DefaultExecutionSpace>::member_type &teamMember) {
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
                Kokkos::TeamThreadRange(teamMember, workSizeI),
                [&](const int j) {
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
              const Kokkos::TeamPolicy<
                  Kokkos::DefaultExecutionSpace>::member_type &teamMember) {
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
                Kokkos::TeamThreadRange(teamMember, workSizeI),
                [&](const int j) {
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

  void FarDot(DeviceFloatMatrix u, DeviceFloatMatrix f) {
    std::cout << "start of FarDot" << std::endl;

    double queryDuration = 0.0;
    double dotDuration = 0.0;
    double ckNormalizationDuration = 0.0;
    double qkNormalizationDuration = 0.0;

    std::size_t totalNumQuery = 0;
    std::size_t totalNumIter = 0;

    const int maxRelativeCoord = 500000;
    const int matPoolSize = maxRelativeCoord * 10;
    const int maxWorkNodeSize = 5000;
    int workNodeSize = 0;
    int cMatPoolUsedSize = 0;
    int qMatPoolUsedSize = 0;
    int allowedWorkload = 0;
    int allowedWorkloadBasedOnMatPool = 0;

    DeviceFloatMatrix relativeCoordPool("relativeCoordPool", maxRelativeCoord,
                                        3);
    DeviceDoubleMatrix cMatPool("cMatPool", matPoolSize, 9);
    DeviceDoubleMatrix qMatPool("qMatPool", matPoolSize, 9);

    DeviceIntVector workingNode("workingNode", maxWorkNodeSize);
    DeviceIntMatrix workingNodeCMatOffset("workingNodeCMatOffset",
                                          maxWorkNodeSize, 100);
    DeviceIntMatrix workingNodeCMatIndex("workingNodeCMatIndex",
                                         maxWorkNodeSize, 100);
    DeviceIntMatrix workingNodeQMatOffset("workingNodeQMatOffset",
                                          maxWorkNodeSize, 100);
    DeviceIntMatrix workingNodeQMatIndex("workingNodeQMatIndex",
                                         maxWorkNodeSize, 100);
    DeviceIntVector workingNodeIteration("workingNodeIteration",
                                         maxWorkNodeSize);

    DeviceIntVector relativeCoordSize("relativeCoordSize", maxWorkNodeSize);
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
      if (workNodeSize == 0) {
        // select working node
        allowedWorkloadBasedOnMatPool = std::min(
            matPoolSize - cMatPoolUsedSize, matPoolSize - qMatPoolUsedSize);
        allowedWorkload =
            std::min(allowedWorkloadBasedOnMatPool, maxRelativeCoord);

        // estimate the workload
        int estimatedWorkload;
        int oldWorkNodeSize = workNodeSize;
        int leftNode =
            std::min(farNodeSize - finishedNodeSize, maxWorkNodeSize);
        workNodeSize = oldWorkNodeSize + leftNode;
        int lowerWorkNodeSize = oldWorkNodeSize;
        int upperWorkNodeSize = workNodeSize;
        // install new working node
        Kokkos::parallel_for(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                0, std::min(leftNode, maxWorkNodeSize)),
            KOKKOS_LAMBDA(const std::size_t i) {
              workingNode(i + oldWorkNodeSize) = installedNode + i;
            });
        while (true) {
          int estimatedQMatWorkload = 0;
          int estimatedCMatWorkload = 0;

          Kokkos::parallel_reduce(
              Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,
                                                                 workNodeSize),
              KOKKOS_LAMBDA(const std::size_t i, int &tSum) {
                int nodeI = mFarMatI(workingNode(i));
                int workload = mClusterTree(nodeI, 3) - mClusterTree(nodeI, 2);
                tSum += workload;
              },
              Kokkos::Sum<int>(estimatedCMatWorkload));

          Kokkos::parallel_reduce(
              Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,
                                                                 workNodeSize),
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

        installedNode += workNodeSize - oldWorkNodeSize;

        printf("\r%d nodes installed, %d nodes working", installedNode,
               workNodeSize);

        Kokkos::parallel_for(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workNodeSize),
            KOKKOS_LAMBDA(const std::size_t i) {
              workingNodeIteration(i) = 0;
              workingNodeCMatIndex(i, 0) = 0;
            });
      }

      {
        // calculate relative coord for C
        totalCoord = 0;
        Kokkos::parallel_reduce(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workNodeSize),
            KOKKOS_LAMBDA(const std::size_t i, int &tSum) {
              const int nodeI = mFarMatI(workingNode(i));

              const int indexIStart = mClusterTree(nodeI, 2);
              const int indexIEnd = mClusterTree(nodeI, 3);

              const int workSizeI = indexIEnd - indexIStart;

              relativeCoordSize(i) = workSizeI;

              tSum += workSizeI;
            },
            Kokkos::Sum<int>(totalCoord));
        Kokkos::fence();

        totalNumQuery += totalCoord;
        Kokkos::parallel_for(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workNodeSize),
            KOKKOS_LAMBDA(const int rank) {
              relativeCoordOffset(rank) = 0;
              for (int i = 0; i < rank; i++) {
                relativeCoordOffset(rank) += relativeCoordSize(i);
              }
            });
        Kokkos::fence();

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
                  workingNodeCMatIndex(rank, workingNodeIteration(rank));

              Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(teamMember, workSizeI),
                  [&](const int j) {
                    const int index = relativeOffset + j;
                    for (int l = 0; l < 3; l++) {
                      relativeCoordPool(index, l) =
                          mCoord(indexJ, l) - mCoord(indexIStart + j, l);
                    }
                  });
            });
        Kokkos::fence();

        // do inference for CMat
        auto options = torch::TensorOptions()
                           .dtype(torch::kFloat32)
                           .device(torch::kCUDA, 0)
                           .requires_grad(false);
        torch::Tensor relativeCoordTensor = torch::from_blob(
            relativeCoordPool.data(), {totalCoord, 3}, options);
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

              const int nodeJ = mFarMatJ(workingNode(rank));
              const int indexJ =
                  mClusterTree(nodeJ, 2) +
                  workingNodeCMatIndex(rank, workingNodeIteration(rank));

              Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(teamMember, workSizeI),
                  [&](const int j) {
                    const int index = relativeOffset + j;
                    for (int l = 0; l < 9; l++)
                      cMatPool(cMatPoolUsedSize + index, l) =
                          dataPtr[9 * index + l];
                  });
            });
        Kokkos::fence();

        Kokkos::parallel_for(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workNodeSize),
            KOKKOS_LAMBDA(const int rank) {
              const int relativeOffset = relativeCoordOffset(rank);
              workingNodeCMatOffset(rank, workingNodeIteration(rank)) =
                  cMatPoolUsedSize + relativeOffset;
            });
        Kokkos::fence();

        cMatPoolUsedSize += totalCoord;
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
              // normalize the CMat

              const int nodeI = mFarMatI(workingNode(rank));
              const int indexIStart = mClusterTree(nodeI, 2);
              const int indexIEnd = mClusterTree(nodeI, 3);

              const int rowSize = indexIEnd - indexIStart;

              const int ckOffset =
                  workingNodeCMatOffset(rank, workingNodeIteration(rank));
              const int jk =
                  workingNodeCMatIndex(rank, workingNodeIteration(rank));

              for (int l = 0; l < workingNodeIteration(rank); l++) {
                const int cMatOffsetL = workingNodeCMatOffset(rank, l);
                const int qMatOffsetJk = workingNodeQMatOffset(rank, l) + jk;

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(teamMember, rowSize),
                    [&](const int j) {
                      const int index = ckOffset + j;
                      const int indexL = cMatOffsetL + j;

                      for (int row = 0; row < 3; row++) {
                        cMatPool(index, 3 * row) -=
                            cMatPool(indexL, 3 * row) *
                                qMatPool(qMatOffsetJk, 0) +
                            cMatPool(indexL, 3 * row + 1) *
                                qMatPool(qMatOffsetJk, 3) +
                            cMatPool(indexL, 3 * row + 2) *
                                qMatPool(qMatOffsetJk, 6);

                        cMatPool(index, 3 * row + 1) -=
                            cMatPool(indexL, 3 * row) *
                                qMatPool(qMatOffsetJk, 1) +
                            cMatPool(indexL, 3 * row + 1) *
                                qMatPool(qMatOffsetJk, 4) +
                            cMatPool(indexL, 3 * row + 2) *
                                qMatPool(qMatOffsetJk, 7);

                        cMatPool(index, 3 * row + 2) -=
                            cMatPool(indexL, 3 * row) *
                                qMatPool(qMatOffsetJk, 2) +
                            cMatPool(indexL, 3 * row + 1) *
                                qMatPool(qMatOffsetJk, 5) +
                            cMatPool(indexL, 3 * row + 2) *
                                qMatPool(qMatOffsetJk, 8);
                      }
                    });
              }
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

              Kokkos::MaxLoc<double, int>::value_type result;
              Kokkos::parallel_reduce(
                  Kokkos::TeamThreadRange(teamMember, rowSize),
                  [&](const int j,
                      Kokkos::MaxLoc<double, int>::value_type &update) {
                    double curNorm = 0.0;
                    const int index = ckOffset + j;
                    for (int row = 0; row < 3; row++)
                      for (int col = 0; col < 3; col++)
                        curNorm += pow(cMatPool(index, 3 * row + col), 2);

                    if (curNorm > update.val) {
                      update.val = curNorm;
                      update.loc = j;
                    }
                  },
                  Kokkos::MaxLoc<double, int>(result));

              workingNodeQMatIndex(rank, workingNodeIteration(rank)) =
                  result.loc;
            });
        Kokkos::fence();

        auto end = std::chrono::steady_clock::now();
        ckNormalizationDuration +=
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
                .count();
      }

      {
        // calculate relative coord for Q
        totalCoord = 0;
        Kokkos::parallel_reduce(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workNodeSize),
            KOKKOS_LAMBDA(const std::size_t i, int &tSum) {
              const int nodeJ = mFarMatJ(workingNode(i));

              const int indexJStart = mClusterTree(nodeJ, 2);
              const int indexJEnd = mClusterTree(nodeJ, 3);

              const int workSizeJ = indexJEnd - indexJStart;

              relativeCoordSize(i) = workSizeJ;

              tSum += workSizeJ;
            },
            Kokkos::Sum<int>(totalCoord));
        Kokkos::fence();

        totalNumQuery += totalCoord;
        Kokkos::parallel_for(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workNodeSize),
            KOKKOS_LAMBDA(const int rank) {
              relativeCoordOffset(rank) = 0;
              for (int i = 0; i < rank; i++) {
                relativeCoordOffset(rank) += relativeCoordSize(i);
              }
            });
        Kokkos::fence();

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
                  workingNodeQMatIndex(rank, workingNodeIteration(rank));

              Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(teamMember, workSizeJ),
                  [&](const int j) {
                    const int index = relativeOffset + j;
                    for (int l = 0; l < 3; l++) {
                      relativeCoordPool(index, l) =
                          mCoord(indexI, l) - mCoord(indexJStart + j, l);
                    }
                  });
            });
        Kokkos::fence();

        // do inference for QMat
        auto options = torch::TensorOptions()
                           .dtype(torch::kFloat32)
                           .device(torch::kCUDA, 0)
                           .requires_grad(false);
        torch::Tensor relativeCoordTensor = torch::from_blob(
            relativeCoordPool.data(), {totalCoord, 3}, options);
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

              const int nodeI = mFarMatI(workingNode(rank));

              Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(teamMember, workSizeJ),
                  [&](const int j) {
                    const int index = relativeOffset + j;
                    for (int l = 0; l < 9; l++)
                      qMatPool(qMatPoolUsedSize + index, l) =
                          dataPtr[9 * index + l];
                  });
            });
        Kokkos::fence();

        Kokkos::parallel_for(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workNodeSize),
            KOKKOS_LAMBDA(const int rank) {
              const int relativeOffset = relativeCoordOffset(rank);
              workingNodeQMatOffset(rank, workingNodeIteration(rank)) =
                  qMatPoolUsedSize + relativeOffset;
            });
        Kokkos::fence();

        qMatPoolUsedSize += totalCoord;
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
              // normalize the QMat

              const int nodeI = mFarMatI(workingNode(rank));
              const int indexIStart = mClusterTree(nodeI, 2);
              const int indexIEnd = mClusterTree(nodeI, 3);

              const int rowSize = indexIEnd - indexIStart;

              const int qkOffset =
                  workingNodeQMatOffset(rank, workingNodeIteration(rank));
              const int ik =
                  workingNodeQMatIndex(rank, workingNodeIteration(rank));

              for (int l = 0; l < workingNodeIteration(rank); l++) {
                const int qMatOffsetL = workingNodeQMatOffset(rank, l);
                const int cMatOffsetIk = workingNodeQMatOffset(rank, l) + ik;

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(teamMember, rowSize),
                    [&](const int j) {
                      const int index = qkOffset + j;
                      const int indexL = qMatOffsetL + j;

                      for (int row = 0; row < 3; row++) {
                        qMatPool(index, 3 * row) -=
                            cMatPool(cMatOffsetIk, 3 * row) *
                                qMatPool(indexL, 3 * row) +
                            cMatPool(cMatOffsetIk, 3 * row + 1) *
                                qMatPool(indexL, 3 * row + 1) +
                            cMatPool(cMatOffsetIk, 3 * row + 2) *
                                qMatPool(indexL, 3 * row + 2);
                      }
                    });
              }
            });
        Kokkos::fence();
        auto end = std::chrono::steady_clock::now();

        qkNormalizationDuration +=
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
                .count();
      }

      // stop criterion
      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workNodeSize),
          KOKKOS_LAMBDA(const int i) {
            workingNodeIteration(i) += 1;
            workingNodeCMatIndex(i, workingNodeIteration(i)) = 0;
            if (workingNodeIteration(i) == 8) {
              workingNode(i) = -1;
            }
          });

      int finishedWorkNodeSize = 0;
      Kokkos::parallel_reduce(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workNodeSize),
          KOKKOS_LAMBDA(const std::size_t i, int &tFinishedWorkNodeSize) {
            if (workingNode(i) == -1) {
              tFinishedWorkNodeSize += 1;
            }
          },
          Kokkos::Sum<int>(finishedWorkNodeSize));

      if (finishedWorkNodeSize == workNodeSize) {
        finishedNodeSize += workNodeSize;
        workNodeSize = 0;
        cMatPoolUsedSize = 0;
        qMatPoolUsedSize = 0;
      }

      if (finishedNodeSize == farNodeSize) {
        break;
      }
    }

    std::cout << std::endl;

    printf("num query: %ld, num iteration: %ld, query duration: %.4fs, dot "
           "duration: %.4fs\n",
           totalNumQuery, totalNumIter, queryDuration / 1e6, dotDuration / 1e6);
    printf(
        "ck normalization duration: %.4fs, qk normalization duration: %.4fs\n",
        ckNormalizationDuration / 1e6, qkNormalizationDuration / 1e6);

    std::cout << "end of FarDot" << std::endl;
  }

  void Dot(DeviceFloatMatrix u, DeviceFloatMatrix f) {
    std::cout << "start of Dot" << std::endl;

    // initialize u
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, u.extent(0)),
        KOKKOS_LAMBDA(const int i) {
          u(i, 0) = 0.0;
          u(i, 1) = 0.0;
          u(i, 2) = 0.0;
        });
    Kokkos::fence();

    CloseDot(u, f);
    FarDot(u, f);

    std::cout << "end of Dot" << std::endl;
  }
};

#endif