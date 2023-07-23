#ifndef HIGNN_HPP
#define HIGNN_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

#include <mpi.h>

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

  std::vector<std::size_t> mLeafNodeList;

  std::vector<std::size_t> mReorderedMap;

  int mBlockSize;
  int mDim;

  int mMPIRank;
  int mMPISize;

  std::size_t clusterTreeSize;
  torch::jit::script::Module model;

  std::string mDeviceString;

protected:
  std::size_t GetCount() {
    return mCoordPtr->extent(0);
  }

  void ComputeAux(const std::size_t first,
                  const std::size_t last,
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

  std::size_t Divide(const std::size_t first,
                     const std::size_t last,
                     const int axis,
                     std::vector<std::size_t> &reorderedMap) {
    auto &mVertexMirror = *mCoordMirrorPtr;

    const std::size_t L = last - first;

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

    MPI_Comm_rank(MPI_COMM_WORLD, &mMPIRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mMPISize);

    // load script model
    model = torch::jit::load("3D_force_UB_max600_try2.pt");
    mDeviceString = "cuda:" + std::to_string(mMPIRank);
    model.to(mDeviceString);

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(torch::kCUDA, mMPIRank)
                       .requires_grad(false);
    torch::Tensor testTensor = torch::ones({50000, 3}, options);
    std::vector<c10::IValue> inputs;
    inputs.push_back(testTensor);

    auto testResult = model.forward(inputs);
  }

  void Update() {
    Build();

    CloseFarCheck();
  }

  void Build() {
    MPI_Barrier(MPI_COMM_WORLD);
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();

    if (mMPIRank == 0)
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

    MPI_Barrier(MPI_COMM_WORLD);
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    if (mMPIRank == 0) {
      std::cout << "Build time: " << (double)duration / 1e6 << "s" << std::endl;

      std::cout << "end of Build" << std::endl;
    }
  }

  inline bool CloseFarCheck(HostFloatMatrix aux,
                            const std::size_t node1,
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

    if (mMPIRank == 0)
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

            std::size_t nodeSizeI =
                mClusterTree(node, 3) - mClusterTree(node, 2);
            std::size_t nodeSizeJ = mClusterTree(closeMat[i][j], 3) -
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

          std::size_t nodeSizeI = mClusterTree(node, 3) - mClusterTree(node, 2);
          std::size_t nodeSizeJ =
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

        mLeafNodeList.push_back(node);
      }
    }

    if (mMPIRank == 0) {
      std::cout << "total entry: " << total_entry << std::endl;

      std::cout << "num of leaf nodes: " << mLeafNodeList.size() << std::endl;
    }

    // split leaf node among mpi ranks
    std::size_t leafNodeStart, leafNodeStartI, leafNodeStartJ;
    std::size_t leafNodeEnd, leafNodeEndI, leafNodeEndJ;

    std::size_t totalCloseMatSize = 0;
    for (int i = 0; i < mLeafNodeList.size(); i++)
      totalCloseMatSize += closeMat[mLeafNodeList[i]].size();

    std::size_t estimatedLeafNodeSize =
        std::ceil((double)totalCloseMatSize / mMPISize);
    leafNodeStart = estimatedLeafNodeSize * mMPIRank;
    leafNodeEnd =
        std::min(estimatedLeafNodeSize * (mMPIRank + 1), totalCloseMatSize);

    int counter = 0;
    for (int i = 0; i < mLeafNodeList.size(); i++) {
      counter += closeMat[mLeafNodeList[i]].size();
      if (leafNodeStart <= counter) {
        leafNodeStartI = i;
        leafNodeStartJ =
            leafNodeStart - (counter - closeMat[mLeafNodeList[i]].size());
        break;
      }
    }

    counter = 0;
    for (int i = 0; i < mLeafNodeList.size(); i++) {
      counter += closeMat[mLeafNodeList[i]].size();
      if (leafNodeEnd <= counter) {
        leafNodeEndI = i;
        leafNodeEndJ =
            leafNodeEnd - (counter - closeMat[mLeafNodeList[i]].size());
        break;
      }
    }
    std::size_t leafNodeSize = leafNodeEndI - leafNodeStartI + 1;

    mLeafNodePtr = std::make_shared<DeviceIndexVector>();
    Kokkos::resize(*mLeafNodePtr, leafNodeSize);

    DeviceIndexVector::HostMirror hostLeafNode =
        Kokkos::create_mirror_view(*mLeafNodePtr);

    auto &mLeafNode = *mLeafNodePtr;

    for (int i = 0; i < leafNodeSize; i++)
      hostLeafNode(i) = mLeafNodeList[i + leafNodeStartI];
    Kokkos::deep_copy(mLeafNode, hostLeafNode);

    mCloseMatIPtr = std::make_shared<DeviceIndexVector>();
    auto &mCloseMatI = *mCloseMatIPtr;
    Kokkos::resize(*mCloseMatIPtr, leafNodeSize + 1);

    DeviceIndexVector::HostMirror hostCloseMatI =
        Kokkos::create_mirror_view(*mCloseMatIPtr);
    hostCloseMatI(0) = 0;
    for (int i = 0; i < leafNodeSize; i++) {
      if (i == 0)
        hostCloseMatI(i + 1) =
            closeMat[mLeafNodeList[i + leafNodeStartI]].size() - leafNodeStartJ;
      else if (i == leafNodeSize - 1)
        hostCloseMatI(i + 1) = hostCloseMatI(i) + leafNodeEndJ;
      else
        hostCloseMatI(i + 1) =
            hostCloseMatI(i) +
            closeMat[mLeafNodeList[i + leafNodeStartI]].size();
    }

    mCloseMatJPtr = std::make_shared<DeviceIndexVector>();
    auto &mCloseMatJ = *mCloseMatJPtr;
    Kokkos::resize(*mCloseMatJPtr, hostCloseMatI(leafNodeSize));

    DeviceIndexVector::HostMirror hostCloseMatJ =
        Kokkos::create_mirror_view(*mCloseMatJPtr);
    for (int i = 0; i < leafNodeSize; i++) {
      if (i == 0)
        for (int j = leafNodeStartJ;
             j < closeMat[mLeafNodeList[i + leafNodeStartI]].size(); j++)
          hostCloseMatJ(hostCloseMatI(i) + j - leafNodeStartJ) =
              closeMat[mLeafNodeList[i + leafNodeStartI]][j];
      else if (i == leafNodeSize - 1)
        for (int j = 0; j < leafNodeEndJ; j++)
          hostCloseMatJ(hostCloseMatI(i) + j) =
              closeMat[mLeafNodeList[i + leafNodeStartI]][j];
      else
        for (int j = 0; j < closeMat[mLeafNodeList[i + leafNodeStartI]].size();
             j++)
          hostCloseMatJ(hostCloseMatI(i) + j) =
              closeMat[mLeafNodeList[i + leafNodeStartI]][j];
    }

    if (mMPIRank == 0)
      std::cout << "Total close pair: " << hostCloseMatI(mLeafNodeList.size())
                << std::endl;

    std::size_t totalCloseEntry = 0;
    for (int i = 0; i < mLeafNodeList.size(); i++) {
      int nodeI = mLeafNodeList[i];
      int nodeSizeI = mClusterTree(nodeI, 3) - mClusterTree(nodeI, 2);
      for (int j = 0; j < closeMat[nodeI].size(); j++) {
        int nodeJ = closeMat[nodeI][j];
        int nodeSizeJ = mClusterTree(nodeJ, 3) - mClusterTree(nodeJ, 2);

        totalCloseEntry += nodeSizeI * nodeSizeJ;
      }
    }

    if (mMPIRank == 0)
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
    if (mMPIRank == 0) {
      std::cout << "Total far node: " << totalFarNode << std::endl;
      std::cout << "Total far pair: " << totalFarSize << std::endl;
    }

    // split far pair among mpi ranks
    std::size_t farMatSize = totalFarSize / (std::size_t)mMPISize;
    farMatSize += ((totalFarSize % (std::size_t)mMPISize) > mMPIRank) ? 1 : 0;

    mFarMatIPtr = std::make_shared<DeviceIndexVector>();
    auto &mFarMatI = *mFarMatIPtr;
    Kokkos::resize(*mFarMatIPtr, farMatSize);

    mFarMatJPtr = std::make_shared<DeviceIndexVector>();
    auto &mFarMatJ = *mFarMatJPtr;
    Kokkos::resize(*mFarMatJPtr, farMatSize);

    DeviceIndexVector::HostMirror farMatIMirror =
        Kokkos::create_mirror_view(mFarMatI);
    DeviceIndexVector::HostMirror farMatJMirror =
        Kokkos::create_mirror_view(mFarMatJ);

    counter = 0;
    int matCounter = 0;
    for (int i = 0; i < farMat.size(); i++) {
      for (int j = 0; j < farMat[i].size(); j++) {
        if (counter % mMPISize == mMPIRank) {
          farMatIMirror(matCounter) = i;
          farMatJMirror(matCounter) = farMat[i][j];
          matCounter++;
        }
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
    MPI_Allreduce(MPI_IN_PLACE, &farDotQueryNum, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                  MPI_COMM_WORLD);
    if (mMPIRank == 0)
      std::cout << "Total far entry: " << farDotQueryNum << std::endl;

    Kokkos::deep_copy(mFarMatI, farMatIMirror);
    Kokkos::deep_copy(mFarMatJ, farMatJMirror);

    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    if (mMPIRank == 0) {
      std::cout << "Time for building close and far matrix: "
                << (double)duration / 1e6 << "s" << std::endl;
    }
  }

  void PostCheck();

  void PostCheckDot(DeviceDoubleMatrix u, DeviceDoubleMatrix f);

  void CloseDot(DeviceDoubleMatrix u, DeviceDoubleMatrix f);

  void FarDot(DeviceDoubleMatrix u, DeviceDoubleMatrix f);

  void Dot(DeviceDoubleMatrix u, DeviceDoubleMatrix f) {
    if (mMPIRank == 0)
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

    DeviceDoubleMatrix::HostMirror hostU = Kokkos::create_mirror_view(u);
    Kokkos::deep_copy(hostU, u);

    MPI_Allreduce(MPI_IN_PLACE, hostU.data(), hostU.extent(0) * hostU.extent(1),
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    Kokkos::deep_copy(u, hostU);

    // PostCheckDot(u, f);
    // PostCheck();

    if (mMPIRank == 0)
      std::cout << "end of Dot" << std::endl;
  }
};

#endif