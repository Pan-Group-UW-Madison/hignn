#include <execution>

#include "HignnModel.hpp"

using namespace std;
using namespace std::chrono;

void Init() {
  int flag;

  MPI_Initialized(&flag);

  if (!flag)
    MPI_Init(NULL, NULL);

  int mpiRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

#ifdef USE_GPU
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  auto settings = Kokkos::InitializationSettings()
                      .set_num_threads(10)
                      .set_device_id(mpiRank % deviceCount)
                      .set_disable_warnings(false);
#else
  auto settings =
      Kokkos::InitializationSettings().set_num_threads(10).set_disable_warnings(
          false);
#endif

  Kokkos::initialize(settings);
}

void Finalize() {
  Kokkos::finalize();

  // MPI_Finalize();
}

size_t HignnModel::GetCount() {
  return mCoordPtr->extent(0);
}

void HignnModel::ComputeAux(const std::size_t first,
                            const std::size_t last,
                            std::vector<float> &aux) {
  auto &mVertexMirror = *mCoordMirrorPtr;

  for (int d = 0; d < mDim; d++) {
    aux[2 * d] = std::numeric_limits<float>::max();
    aux[2 * d + 1] = -std::numeric_limits<float>::max();
  }

  // Iterate over the specified range and update min/max.
  for (size_t i = first; i < last; i++)
    for (int d = 0; d < mDim; d++) {
      aux[2 * d] =
          (aux[2 * d] > mVertexMirror(i, d)) ? mVertexMirror(i, d) : aux[2 * d];
      aux[2 * d + 1] = (aux[2 * d + 1] < mVertexMirror(i, d))
                           ? mVertexMirror(i, d)
                           : aux[2 * d + 1];
    }
}

size_t HignnModel::Divide(const std::size_t first,
                          const std::size_t last,
                          std::vector<std::size_t> &reorderedMap,
                          const bool parallelFlag) {
  auto &mVertexMirror = *mCoordMirrorPtr;

  const std::size_t L = last - first;

  std::vector<float> mean(mDim, 0.0);
  std::vector<float> temp(L);

  for (size_t i = first; i < last; i++)
    for (int d = 0; d < mDim; d++)
      mean[d] += mVertexMirror(reorderedMap[i], d);
  for (int d = 0; d < mDim; d++)
    mean[d] /= (float)L;

  Eigen::MatrixXf vertexHat(mDim, L);
  for (int d = 0; d < mDim; d++) {
    if (parallelFlag)
#pragma omp parallel for schedule(static, 1024)
      for (size_t i = 0; i < L; i++) {
        vertexHat(d, i) = mVertexMirror(reorderedMap[i + first], d) - mean[d];
      }
    else
      for (size_t i = 0; i < L; i++) {
        vertexHat(d, i) = mVertexMirror(reorderedMap[i + first], d) - mean[d];
      }
  }

  Eigen::JacobiSVD<Eigen::MatrixXf> svdHolder(
      vertexHat, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::MatrixXf U = svdHolder.matrixU();

  for (size_t i = 0; i < L; i++) {
    temp[i] = 0.0;
    for (int d = 0; d < mDim; d++) {
      temp[i] += U(d, 0) * vertexHat(d, i);
    }
  }

  std::vector<std::size_t> newIndex;
  newIndex.resize(temp.size());
  iota(newIndex.begin(), newIndex.end(), 0);

  if (parallelFlag) {
    sort(std::execution::par_unseq, newIndex.begin(), newIndex.end(),
         [&temp](int i1, int i2) { return temp[i1] < temp[i2]; });
    sort(std::execution::par_unseq, temp.begin(), temp.end());
  } else {
    sort(newIndex.begin(), newIndex.end(),
         [&temp](int i1, int i2) { return temp[i1] < temp[i2]; });
    sort(temp.begin(), temp.end());
  }

  // Currently, the reordering is a very stupid implementation. Need to
  // improve it.

  std::vector<std::size_t> copyIndex(L);

  if (parallelFlag) {
#pragma omp parallel for schedule(static, 1024)
    for (size_t i = 0; i < L; i++)
      copyIndex[i] = reorderedMap[i + first];

#pragma omp parallel for schedule(static, 1024)
    for (size_t i = 0; i < L; i++)
      reorderedMap[i + first] = copyIndex[newIndex[i]];
  } else {
    for (size_t i = 0; i < L; i++)
      copyIndex[i] = reorderedMap[i + first];

    for (size_t i = 0; i < L; i++)
      reorderedMap[i + first] = copyIndex[newIndex[i]];
  }

  auto result = std::upper_bound(temp.begin(), temp.end(), 0);

  return (std::size_t)(result - temp.begin()) + first;
}

void HignnModel::Reorder(const std::vector<std::size_t> &reorderedMap) {
  auto &mVertexMirror = *mCoordMirrorPtr;
  auto &mVertex = *mCoordPtr;

  // Create a temporary copy of current coordinates.
  HostFloatMatrix copyVertex;
  Kokkos::resize(copyVertex, reorderedMap.size(), 3);

  // Copy original coordinates to the temporary buffer.
  for (size_t i = 0; i < reorderedMap.size(); i++) {
    copyVertex(i, 0) = mVertexMirror(i, 0);
    copyVertex(i, 1) = mVertexMirror(i, 1);
    copyVertex(i, 2) = mVertexMirror(i, 2);
  }

  // Rearrange coordinates in mVertexMirror according to reorderedMap.
  for (size_t i = 0; i < reorderedMap.size(); i++) {
    mVertexMirror(i, 0) = copyVertex(reorderedMap[i], 0);
    mVertexMirror(i, 1) = copyVertex(reorderedMap[i], 1);
    mVertexMirror(i, 2) = copyVertex(reorderedMap[i], 2);
  }

  // Copy reordered coordinates from host to device
  Kokkos::deep_copy(mVertex, mVertexMirror);
}

void HignnModel::Reorder(const std::vector<size_t> &reorderedMap,
                         DeviceDoubleMatrix v) {
  // Create a device matrix for the reordered result
  DeviceDoubleMatrix vCopy("vCopy", v.extent(0), v.extent(1));

  // Transfer the reorderedMap to device memory
  DeviceIndexVector deviceReorderedMap("reorderedMap", reorderedMap.size());
  auto hostReorderedMap = Kokkos::create_mirror_view(deviceReorderedMap);

  for (size_t i = 0; i < reorderedMap.size(); i++)
    hostReorderedMap(i) = reorderedMap[i];

  Kokkos::deep_copy(deviceReorderedMap, hostReorderedMap);

  // Rearrange rows of v on device according to reorderedMap
  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, v.extent(0)),
      KOKKOS_LAMBDA(const int i) {
        vCopy(i, 0) = v(deviceReorderedMap(i), 0);
        vCopy(i, 1) = v(deviceReorderedMap(i), 1);
        vCopy(i, 2) = v(deviceReorderedMap(i), 2);
      });

  // Copy reordered matrix back into original v
  Kokkos::deep_copy(v, vCopy);
}

void HignnModel::BackwardReorder(const std::vector<size_t> &reorderedMap,
                                 DeviceDoubleMatrix v) {
  // Create a temporary copy of v with the same dimensions
  DeviceDoubleMatrix vCopy("vCopy", v.extent(0), v.extent(1));

  // Prepare a device vector storing the reordered indices
  DeviceIndexVector deviceReorderedMap("reorderedMap", reorderedMap.size());
  auto hostReorderedMap = Kokkos::create_mirror_view(deviceReorderedMap);

  // Copy the reorderedMap indices to the Kokkos host view
  for (size_t i = 0; i < reorderedMap.size(); i++)
    hostReorderedMap(i) = reorderedMap[i];

  // Copy the index mapping from host to device
  Kokkos::deep_copy(deviceReorderedMap, hostReorderedMap);

  // Perform the actual reordering in parallel on the device
  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, v.extent(0)),
      KOKKOS_LAMBDA(const int i) {
        vCopy(deviceReorderedMap(i), 0) = v(i, 0);
        vCopy(deviceReorderedMap(i), 1) = v(i, 1);
        vCopy(deviceReorderedMap(i), 2) = v(i, 2);
      });

  // Copy the reordered result back into v
  Kokkos::deep_copy(v, vCopy);
}

HignnModel::HignnModel(pybind11::array_t<float> &coord, const int blockSize) {
  // default values
  mPostCheckFlag = false;
  mUseSymmetry = true;

  mMaxFarDotWorkNodeSize = 5000;

  mMaxRelativeCoord = 500000;

  mMaxFarFieldDistance = 1000;

  // Get shape and access data from numpy array (num_particles × 3)
  auto data = coord.unchecked<2>();
  auto shape = coord.shape();

  // Allocate device and host memory for particle coordinates
  mCoordPtr = std::make_shared<DeviceFloatMatrix>(
      DeviceFloatMatrix("mCoordPtr", (size_t)shape[0], (size_t)shape[1]));
  mCoordMirrorPtr = std::make_shared<DeviceFloatMatrix::HostMirror>();
  *mCoordMirrorPtr = Kokkos::create_mirror_view(*mCoordPtr);

  // Fill the host coordinates with data from the input numpy array
  auto &hostCoord = *mCoordMirrorPtr;

  for (size_t i = 0; i < (size_t)shape[0]; i++) {
    hostCoord(i, 0) = data(i, 0);
    hostCoord(i, 1) = data(i, 1);
    hostCoord(i, 2) = data(i, 2);
  }

  // Copy the initialized host coordinates to device memory
  Kokkos::deep_copy(*mCoordPtr, hostCoord);

  // Store user-specified block size and set model to 3D
  mBlockSize = blockSize;
  mDim = 3;

  MPI_Comm_rank(MPI_COMM_WORLD, &mMPIRank);
  MPI_Comm_size(MPI_COMM_WORLD, &mMPISize);

  // Initialize default solver/model parameters
  mEpsilon = 0.05;
  mEta = 1.0;
  mMaxIter = 100;
  mMatPoolSizeFactor = 40;

#if USE_GPU
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  mCudaDevice = mMPIRank % deviceCount;
#endif
}

void HignnModel::LoadTwoBodyModel(const std::string &modelPath) {
  // load script model
#if USE_GPU
  mTwoBodyModel =
      torch::jit::load(modelPath + "_" + std::to_string(mCudaDevice) + ".pt");
  mDeviceString = "cuda:" + std::to_string(mCudaDevice);
  mTwoBodyModel.to(mDeviceString);
#else
  mTwoBodyModel = torch::jit::load(modelPath + ".pt");
  mTwoBodyModel.to(torch::kCPU);
#endif

  // default options for torch. As no backward propagation is not required,
  // gradient is not required to be stored and requires_grad is set to false.
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
  // Use a dummy input tensor for model warm-up with a forward pass
  torch::Tensor testTensor = torch::ones({50000, 3}, options);
  std::vector<c10::IValue> inputs;
  inputs.push_back(testTensor);

  auto testResult = mTwoBodyModel.forward(inputs);
}

void HignnModel::LoadThreeBodyModel(
    [[maybe_unused]] const std::string &modelPath) {
}

void HignnModel::Update() {
  Build();

  CloseFarCheck();
}

bool HignnModel::CloseFarCheck(HostFloatMatrix aux,
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

  // Compute squared diameters and center distance for bounding boxes
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

void HignnModel::Dot(pybind11::array_t<float> &uArray,
                     pybind11::array_t<float> &fArray) {
  if (mMPIRank == 0)
    std::cout << "start of Dot" << std::endl;

  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();

  auto shape = fArray.shape();

  DeviceDoubleMatrix u("u", shape[0], 3);
  DeviceDoubleMatrix f("f", shape[0], 3);

  // initialize u
  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, u.extent(0)),
      KOKKOS_LAMBDA(const int i) {
        u(i, 0) = 0.0;
        u(i, 1) = 0.0;
        u(i, 2) = 0.0;
      });
  Kokkos::fence();

  // Access data from Python arrays
  auto fData = fArray.unchecked<2>();
  auto uData = uArray.mutable_unchecked<2>();

  // Host mirror for force array
  DeviceDoubleMatrix::HostMirror hostF = Kokkos::create_mirror_view(f);

  // Copy force data from Python array to host mirror
  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, f.extent(0)),
      [&](const int i) {
        hostF(i, 0) = fData(i, 0);
        hostF(i, 1) = fData(i, 1);
        hostF(i, 2) = fData(i, 2);
      });
  Kokkos::fence();

  // Transfer force data to device memory
  Kokkos::deep_copy(f, hostF);

  // Reorders the force array aligning with the node ordering of the clustering
  // tree.
  Reorder(mReorderedMap, f);

  // Compute close- and far-range velocity contributions
  CloseDot(u, f);
  FarDot(u, f);

  // Copy result velocities back to host
  DeviceDoubleMatrix::HostMirror hostU = Kokkos::create_mirror_view(u);

  Kokkos::deep_copy(hostU, u);

  // Perform an all-reduce to sum the velocity results from all MPI ranks
  MPI_Allreduce(MPI_IN_PLACE, hostU.data(), u.extent(0) * u.extent(1),
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  Kokkos::deep_copy(u, hostU);

  // Optional post-processing check for accuracy
  if (mPostCheckFlag) {
    PostCheckDot(u, f);
  }

  // Restores the velocity array to the original ordering
  BackwardReorder(mReorderedMap, u);

  Kokkos::deep_copy(hostU, u);

  // Copy final velocities to Python output array
  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, u.extent(0)),
      [&](const int i) {
        uData(i, 0) = hostU(i, 0);
        uData(i, 1) = hostU(i, 1);
        uData(i, 2) = hostU(i, 2);
      });
  Kokkos::fence();

  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

  if (mMPIRank == 0)
    printf("End of Dot. Dot time: %.4fs\n", (double)duration / 1e6);
}

void HignnModel::DenseDot(pybind11::array_t<float> &uArray,
                          pybind11::array_t<float> &fArray) {
  auto shape = fArray.shape();

  DeviceDoubleMatrix u("u", shape[0], 3);
  DeviceDoubleMatrix f("f", shape[0], 3);

  // initialize u
  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, u.extent(0)),
      KOKKOS_LAMBDA(const int i) {
        u(i, 0) = 0.0;
        u(i, 1) = 0.0;
        u(i, 2) = 0.0;
      });
  Kokkos::fence();

  // Access data from Python arrays
  auto fData = fArray.unchecked<2>();
  auto uData = uArray.mutable_unchecked<2>();

  // Host mirror for force array
  DeviceDoubleMatrix::HostMirror hostF = Kokkos::create_mirror_view(f);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, f.extent(0)),
      [&](const int i) {
        hostF(i, 0) = fData(i, 0);
        hostF(i, 1) = fData(i, 1);
        hostF(i, 2) = fData(i, 2);
      });
  Kokkos::fence();

  // Transfer force data to device memory
  Kokkos::deep_copy(f, hostF);

  // Reorders the force array aligning with the node ordering of the clustering
  // tree.
  Reorder(mReorderedMap, f);

  // Compute velocities using the dense mobility tensor (on device)
  DenseDot(u, f);

  // Copy velocities to host for MPI reduction
  DeviceDoubleMatrix::HostMirror hostU = Kokkos::create_mirror_view(u);

  Kokkos::deep_copy(hostU, u);

  // Perform an all-reduce to sum the velocity results from all MPI ranks
  MPI_Allreduce(MPI_IN_PLACE, hostU.data(), u.extent(0) * u.extent(1),
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  Kokkos::deep_copy(u, hostU);

  // Restores the velocity array to the original ordering
  BackwardReorder(mReorderedMap, u);

  Kokkos::deep_copy(hostU, u);

  // Copy result velocities to output Python array
  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, u.extent(0)),
      [&](const int i) {
        uData(i, 0) = hostU(i, 0);
        uData(i, 1) = hostU(i, 1);
        uData(i, 2) = hostU(i, 2);
      });
  Kokkos::fence();
}

/**
 * @brief Update the internal coordinates with new particle positions and
 * trigger model update.
 *
 * This function copies the provided coordinate data (shape: num_particles × 3)
 * into the internal host-side coordinate array, then transfers the data to the
 * device-side storage. After updating the coordinates, it calls Update() to
 * rebuild the clustering tree and update all structures dependent on the
 * particle positions.
 *
 * @param coord [in] 2D numpy array of shape (num_particles, 3) containing  the
 * new coordinates for the particles Note: Currently, only nDim = 3 is supported
 * in this implementation.
 */
void HignnModel::UpdateCoord(pybind11::array_t<float> &coord) {
  auto data = coord.unchecked<2>();

  auto shape = coord.shape();

  auto &hostCoord = *mCoordMirrorPtr;

  // Copy each (x, y, z) position from input to host coordinate storage
  for (size_t i = 0; i < (size_t)shape[0]; i++) {
    hostCoord(i, 0) = data(i, 0);
    hostCoord(i, 1) = data(i, 1);
    hostCoord(i, 2) = data(i, 2);
  }

  // Copy updated coordinates from host to device memory
  Kokkos::deep_copy(*mCoordPtr, hostCoord);

  Update();
}

void HignnModel::SetEpsilon(const double epsilon) {
  mEpsilon = epsilon;
}

void HignnModel::SetEta(const double eta) {
  mEta = eta;
}

void HignnModel::SetMaxIter(const int maxIter) {
  mMaxIter = maxIter;
}

void HignnModel::SetMatPoolSizeFactor(const int factor) {
  mMatPoolSizeFactor = factor;
}

void HignnModel::SetPostCheckFlag(const bool flag) {
  mPostCheckFlag = flag;
}

void HignnModel::SetUseSymmetryFlag(const bool flag) {
  mUseSymmetry = flag;
}

void HignnModel::SetMaxFarDotWorkNodeSize(const int size) {
  mMaxFarDotWorkNodeSize = size;
}

void HignnModel::SetMaxRelativeCoord(const size_t size) {
  mMaxRelativeCoord = size;
}

/**
 * @brief Set the far-field cut-off distance for interactions (not used
 * anymore).
 *
 * Updates mMaxFarFieldDistance, the cut-off for far-range interactions.
 *
 * @param distance The cut-off distance.
 */
void HignnModel::SetMaxFarFieldDistance(const double distance) {
  mMaxFarFieldDistance = distance;
}