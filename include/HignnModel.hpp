#ifndef _HignnModel_Hpp_
#define _HignnModel_Hpp_

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
#include <string>

#include <torch/script.h>

using namespace std::chrono;

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>

#include <mpi.h>

#include "Typedef.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

void Init();

void Finalize();

class HignnModel {
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

  unsigned int mBlockSize;
  int mDim;

  int mMPIRank;
  int mMPISize;
#if USE_GPU
  int mCudaDevice;
#endif

  double mEpsilon;

  double mMaxFarFieldDistance;

  int mMaxIter;
  int mMatPoolSizeFactor;

  int mMaxFarDotWorkNodeSize;
  int mMaxCloseDotBlockSize;

  size_t mMaxRelativeCoord;

  std::size_t mClusterTreeSize;
  torch::jit::script::Module mTwoBodyModel;

#if USE_GPU
  std::string mDeviceString;
#endif

  bool mPostCheckFlag;

  bool mUseSymmetry;

protected:
  std::size_t GetCount();

  void ComputeAux(const std::size_t first,
                  const std::size_t last,
                  std::vector<float> &aux);

  std::size_t Divide(const std::size_t first,
                     const std::size_t last,
                     std::vector<std::size_t> &reorderedMap,
                     const bool parallelFlag);

  void Reorder(const std::vector<std::size_t> &reorderedMap);

public:
  HignnModel(pybind11::array_t<float> &coord, const int blockSize);

  void LoadTwoBodyModel(const std::string &modelPath);
  void LoadThreeBodyModel(const std::string &modelPath);

  void Update();

  void Build();

  bool CloseFarCheck(HostFloatMatrix aux,
                     const std::size_t node1,
                     const std::size_t node2);

  void CloseFarCheck();

  void PostCheck();

  void PostCheckDot(DeviceDoubleMatrix u, DeviceDoubleMatrix f);

  void CloseDot(DeviceDoubleMatrix u, DeviceDoubleMatrix f);

  void FarDot(DeviceDoubleMatrix u, DeviceDoubleMatrix f);

  void Dot(pybind11::array_t<float> &uArray, pybind11::array_t<float> &fArray);

  void UpdateCoord(pybind11::array_t<float> &coord);

  void SetEpsilon(const double epsilon);

  void SetMaxIter(const int maxIter);

  void SetMatPoolSizeFactor(const int factor);

  void SetPostCheckFlag(const bool flag);

  void SetUseSymmetryFlag(const bool flag);

  void SetMaxFarDotWorkNodeSize(const int size);

  void SetMaxRelativeCoord(const size_t size);

  void SetMaxFarFieldDistance(const double distance);

  void Reorder(const std::vector<std::size_t> &reorderedMap,
               DeviceDoubleMatrix v);

  void BackwardReorder(const std::vector<std::size_t> &reorderedMap,
                       DeviceDoubleMatrix v);
};

#endif