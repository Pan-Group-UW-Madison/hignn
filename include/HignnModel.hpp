/**
 * @file HignnModel.h
 * @brief The definition of HignnModel class.
 * @author Zisheng Ye
 */

/** \mainpage HIGNN C++ API documentation
 *
 * HIGNN is a framework designed for efficient and scalable simulation of
 * large-scale particulate suspensions. It effectively captures both short- and
 * long-range HIs and their many-body effects and enables substantial
 * computational acceleration by harvesting the power of machine learning and
 * hierarchical matrix.
 *
 * It is developed and maintained by <a href="https://pan.labs.wisc.edu/"
 * >the Pan Group at the University of Wisconsin-Madison</a>
 */

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

/**
 * @brief Initialize MPI (if needed) and Kokkos runtime for parallel
 * computations.
 *
 * Sets up the computational environment for either CPU or GPU based on
 * compile-time flags. This is required before any parallel computation using
 * HignnModel.
 */
void Init();

/**
 * @brief Finalize Kokkos runtime.
 */
void Finalize();

/**
 * @addtogroup hignn_grp HIGNN C++ library
 * @details It contains the main HignnModel class for managing the HIGNN
 * model, including coordinate storage, cluster tree, close/far range
 * interactions, and various settings for running the model.
 *  @{ */

/**
 * @brief It represents the hydrodynamic interaction graph neural
 * network (HIGNN) model.
 *
 * This class manages the coordinates, cluster tree, close/far range
 * interactions, and other settings for running the HIGNN model. It provides
 * methods for building the model, updating the coordinates, and performing
 * various dot operations for calculating particle interactions and velocities.
 */
class HignnModel {
protected:
  std::shared_ptr<DeviceFloatMatrix> mCoordPtr;
  //!< A shared pointer to the matrix storing particle coordinates on the device
  std::shared_ptr<DeviceFloatMatrix::HostMirror> mCoordMirrorPtr;
  //!< A shared pointer to the host-side mirror of the particle coordinates
  //!< matrix.

  std::shared_ptr<DeviceIndexMatrix> mClusterTreePtr;
  //!< A shared pointer to the cluster tree matrix on the device
  std::shared_ptr<DeviceIndexMatrix::HostMirror> mClusterTreeMirrorPtr;
  //!< A shared pointer to the host-side mirror of the cluster tree matrix.

  std::shared_ptr<DeviceIndexVector> mCloseMatIPtr;
  //!< Pointer to device matrix for close-range interaction indices (I).
  std::shared_ptr<DeviceIndexVector> mCloseMatJPtr;
  //!< Pointer to device matrix for close-range interaction indices (J).

  std::shared_ptr<DeviceIndexVector> mFarMatIPtr;
  //!< Pointer to device matrix for far-range interaction indices (I).
  std::shared_ptr<DeviceIndexVector> mFarMatJPtr;
  //!< Pointer to device matrix for far-range interaction indices (J).

  std::shared_ptr<DeviceIndexVector> mLeafNodePtr;
  //!< A shared pointer to the device vector storing leaf node indices.

  std::vector<std::size_t> mLeafNodeList;
  //!< List of all leaf nodes.

  std::vector<std::size_t> mReorderedMap;
  //!< A vector holding the reordered node indices.

  unsigned int mBlockSize;
  //!< Defines the number of nodes processed together in each batch.
  int mDim;
  //!< Dimensionality of the system.

  int mMPIRank;
  int mMPISize;
#if USE_GPU
  int mCudaDevice;
#endif

  double mEpsilon;
  //!< A small value used for numerical stability in calculations.
  double mEta;
  //!< A parameter related to the algorithm's convergence behavior.

  double mMaxFarFieldDistance;
  //!< Maximum distance considered for far-field interactions.

  int mMaxIter;
  //!< Maximum number of iterations for certain operations.

  int mMatPoolSizeFactor;
  //!< Factor controlling the matrix pool size.

  int mMaxFarDotWorkNodeSize;
  //!< Maximum number of nodes to be processed at once in the far dot product.
  int mMaxCloseDotBlockSize;
  //!< Maximum block size for close-range dot product calculations.

  size_t mMaxRelativeCoord;
  //!< Maximum size for storing relative coordinates.

  std::size_t mClusterTreeSize;
  torch::jit::script::Module mTwoBodyModel;
  //!< The two-body interaction model loaded using Torch model.

#if USE_GPU
  std::string mDeviceString;
  //!< String representing the GPU device configuration (if using GPU). This is
  //!< used for loading the right Torch model. When the model is converted by
  //!< python/convert.py, it has already decided the working device. Therefore,
  //!< it has to be converted onto different devices for ensuring load-balance.
#endif

  bool mPostCheckFlag;
  //!< Flag to enable/disable post-checking after computations.

  bool mUseSymmetry;
  //!< Flag to enable/disable the use of symmetry in calculations.

protected:
  /**
   * @brief Get the number of particles in the model.
   *
   * @return size_t The number of particles in the model.
   */
  std::size_t GetCount();

  /**
   * @brief Computes the minimum and maximum values for each dimension (x, y, z)
   *        over a specified range of particle indices.
   *
   * It iterates over the range of particles and updates the
   * auxiliary vector such that for each spatial dimension d, aux[2 * d] stores
   * the minimum and aux[2 * d + 1] stores the maximum coordinate value found in
   * that dimension across all particles in the specified range.
   *
   * @param first The index of the first particle in the range.
   * @param last The index one past the last particle in the range.
   * @param aux A vector of size (2 * mDim) which will be filled with the
   * minimum and maximum values for each spatial dimension.
   */
  void ComputeAux(const std::size_t first,
                  const std::size_t last,
                  std::vector<float> &aux);

  /**
   * @brief Divides particles and reorders them based on their spatial
   * coordinates over a specified range of particle indices.
   *
   * It partitions the specified range of particle indices [first,
   * last) and reorders them based on spatial coordinates using principal
   * component analysis (PCA). The process computes the mean position, centers
   * the coordinates, performs SVD to find the dominant direction, and then
   * sorts the particles along this direction. The particles are thus grouped
   * into clusters for efficient processing. If parallelFlag is true, parts of
   * the computation are performed in parallel for performance.
   *
   * @param first [in] The index of the first particle in the range to consider.
   * @param last [in] The index one past the last particle in the range.
   * @param reorderedMap [in,out] A vector that holds the reordered particle
   * indices after processing. On output, it contains the new ordering for
   * indices in [first, last).
   * @param parallelFlag [in] If true, enables parallel computation where
   * possible.
   *
   * @return The index in reorderedMap where the division ends.
   */
  std::size_t Divide(const std::size_t first,
                     const std::size_t last,
                     std::vector<std::size_t> &reorderedMap,
                     const bool parallelFlag);

  /**
   * @brief Reorders the particle coordinates in both host and device memory
   * based on the provided index mapping.
   *
   * The host-side coordinates are first copied to a temporary buffer, then
   * rearranged according to the order specified in reorderedMap. After
   * reordering, the i-th row of the coordinates corresponds to the particle
   * originally at index reorderedMap[i]. The device-side coordinates are then
   * updated to reflect this new order.
   *
   * @param reorderedMap A vector of size (num_particles) specifying the new
   * order of particle indices.
   */
  void Reorder(const std::vector<std::size_t> &reorderedMap);

public:
  /**
   * @brief Constructor for the HignnModel class.
   *
   * Initializes the model by setting default parameters and allocating memory
   * for the particle coordinates. Sets up the coordinate arrays on both host
   * and device, copies the input coordinates to internal storage, and
   * initializes MPI rank/size and other model parameters.
   *
   * @param coord A 2D numpy array (dimension: num_particles × 3) containing the
   * (x, y, z) positions of all particles.
   * @param blockSize The maximum number of particles allowed in a leaf node (a
   * cluster in the tree that is not further subdivided) during spatial
   * division.
   */
  HignnModel(pybind11::array_t<float> &coord, const int blockSize);

  /**
   * @brief Loads a pre-trained two-body interaction model from the given file
   * path.
   *
   * It loads the model using TorchScript, selecting the appropriate device (CPU
   * or GPU) based on compile-time configuration. For GPU builds, appends the
   * CUDA device ID to the filename to support multi-GPU execution. After
   * loading, the model is moved to the selected device, and a test forward pass
   * is performed with a dummy input tensor of shape (50000, 3) to ensure the
   * model is ready for inference.
   *
   * @param modelPath The base name (without ".pt" extension) of the two-body
   * model file.
   */
  void LoadTwoBodyModel(const std::string &modelPath);

  /**
   * @brief Loads a pre-trained three-body interaction model from the given file
   * path.
   *
   * Currently, three body model is loaded on the python side as no acceleration
   * via C++/Kokkos when doing dot product w.r.t. three-body model.
   *
   * @param modelPath The path to the three-body model file (currently unused).
   */
  void LoadThreeBodyModel(const std::string &modelPath);

  /**
   * @brief Updates the model's state by rebuilding the cluster tree and
   * updating close/far pair information.
   *
   * This method should be called after the coordinates are changed. It rebuilds
   * the internal clustering structure (via Build), then identifies which node
   * pairs are close and which are far for subsequent computations (via
   * CloseFarCheck).
   */
  void Update();

  void Build();

  /**
   * @brief Determines if two nodes are considered 'far' based on bounding boxes
   * and relative distances.
   *
   * Uses the domain of the bounding box of the node and a distance criterion to
   * decide if node1 and node2 should be treated as 'far' pairs, which enables
   * the use of matrix acceleration in subsequent computations. The function
   * first checks if the bounding boxes (defined by minimum and maximum
   * coordinates) of node1 and node2 are disjoint. If they are, the nodes are
   * considered 'far'. Otherwise, it further checks the relative distance and
   * size of their bounding boxes.
   *
   * @param aux      HostFloatMatrix of shape (num_particles, 6), where each row
   * contains the min and max coordinates for x, y, z: [xmin, xmax, ymin, ymax,
   * zmin, zmax].
   * @param node1    Index of the first node (row in aux).
   * @param node2    Index of the second node (row in aux).
   * @return         true if the nodes are considered 'far', false otherwise.
   */
  bool CloseFarCheck(HostFloatMatrix aux,
                     const std::size_t node1,
                     const std::size_t node2);

  /**
   * @brief Splits all of the node pairs in the clustering tree into `close` or
   * `far` pairs.
   *
   * It evaluates whether two sets of particles are considered to be
   * `close` or `far` based on their relative positions of the bounding box of
   * the set. It helps in determining whether the computational model should
   * treat them as interacting closely or not.
   */
  void CloseFarCheck();

  /**
   * @brief Performs a post-processing check after computation.
   *
   * It compares the difference between the dense matrix and the
   * hierarchical matrix acceleration in Frobenius norm. It evaluates the norm
   * by sequentially traversing all of the node pairs in the clustering tree.
   */
  void PostCheck();

  /**
   * @brief Performs a post-processing check after computation.
   *
   * It compares the difference between the dense dot and the
   * hierarchical matrix accelerated dot results in the norm of `u`. The dense
   * dot result of the velocity is obtained within this function using the input
   * force.
   *
   * @param u [in] The velocities obtained by the hierarchical matrix
   * accelerated dot, size (num_particles, 3).
   * @param f [in] The forces, size (num_particles, 3).
   */
  void PostCheckDot(DeviceDoubleMatrix u, DeviceDoubleMatrix f);

  /**
   * @brief Evaluates the updated velocity due to close-range hydrodynamic
   * interactions with the input acting forces.
   *
   * It handles the parallel computation of interactions between node
   * pairs that are marked as `close` on the clustering tree. The workload is
   * divided into smaller batches to save memory usage. Kokkos is used for
   * parallel execution. The function dynamically adjusts the work size based on
   * the estimated workload for each batch.
   *
   * @param u [in, out] A 2D array of size (num_particles, 3) representing the
   * velocities of the particles. The velocities are added with the resulting
   * velocity due to the close-range hydrodynamic interactions w.r.t the acting
   * forces.
   * @param f [in] A 2D array of size (num_particles, 3) representing the forces
   * applied to the particles.
   */
  void CloseDot(DeviceDoubleMatrix u, DeviceDoubleMatrix f);

  /**
   * @brief Evaluates the updated velocity due to far-range hydrodynamic
   *        interactions with the input acting forces.
   *
   * It handles the parallel computation of interactions between node
   * pairs that are marked as `far` on the clustering tree. The workload is
   * divided into adaptive batches to control the maximum memory usage. Kokkos
   * is used for parallel execution. The function dynamically adjusts the batch
   * size based on the estimated workload and then:
   *   - Builds C- and Q-matrices by querying the two-body model with each
   * pair’s relative coordinates.
   *   - Applies a low-rank stopping criterion using an adaptive iterative
   * process inspired by the power-iteration method.
   *   - Accumulates the contributions into the velocity array (u += C·(Q·f)),
   *     with optional symmetry-based updates.
   *
   * @param u [in, out] A 2D array of size (num_particles, 3) representing the
   * velocities of the particles. The velocities are incremented by the
   * resulting velocity contributions due to the far-range hydrodynamic
   * interactions with respect to the acting forces.
   * @param f [in] A 2D array of size (num_particles, 3) representing the forces
   * applied to the particles.
   */
  void FarDot(DeviceDoubleMatrix u, DeviceDoubleMatrix f);

  /**
   * @brief Computes the updated velocities using the original two-body
   * hydrodynamic mobility tensor.
   *
   * It applies the two-body interaction model to all particle
   * pairs, evaluating the hydrodynamic velocities resulting from the provided
   * acting forces. The workload is divided among MPI ranks and further
   * parallelized using Kokkos. All pairwise interactions are processed, without
   * distinguishing between 'close' or 'far' nodes, resulting in dense
   * evaluation of the mobility tensor. The function dynamically adapts the work
   * size and batches to control memory usage, and leverages the TorchScript
   * model for inference on each pair’s relative coordinates.
   *
   * @param u [in, out] A matrix of size (num_particles, 3) representing the
   * velocities of the particles. The velocities are calculated from all
   * pairwise hydrodynamic interactions with respect to the acting forces.
   * @param f [in] A matrix of size (num_particles, 3) representing the forces
   * applied to the particles.
   */
  void DenseDot(DeviceDoubleMatrix u, DeviceDoubleMatrix f);

  /**
   * @brief Computes hydrodynamic interaction and update the velocities from the
   * given forces using hierarchical matrix acceleration to the mobility tensor.
   *
   * It calculates the particle velocities (mobility problem) by
   * applying the hierarchical matrix acceleration when calculating the product
   * between the mobility tensor and the input forces. It supports distributed
   * execution based on MPI and optionally runs a post-processing check. The
   * function performs the following steps:
   *   - Initializes local velocity and force arrays (both with dimensions
   * [num_particles, 3]).
   *   - Copies force data from Python array into device memory.
   *   - Reorders the force array aligning with the node ordering of the
   * clustering tree.
   *   - Computes the close- and far-field contributions to the velocities.
   *   - Collects and sums the results across all MPI ranks.
   *   - Optionally verifies the result if post-check is enabled.
   *   - Restores the velocity array to the original ordering.
   *   - Copies the computed velocities back to the output Python array.
   *
   * @param uArray [out] Output array (num_particles, 3) to be filled with
   * computed velocities for each particle.
   * @param fArray [in]  Input array (num_particles, 3) of forces acting on each
   * particle.
   */
  void Dot(pybind11::array_t<float> &uArray, pybind11::array_t<float> &fArray);

  /**
   * @brief Computes hydrodynamic interaction and update the velocities from the
   * given forces using the dense mobility tensor.
   *
   * It applies the dense (without hierarchical matrix acceleration)
   * mobility tensor to the input force array to compute the resulting
   * velocities. The result is aggregated across all MPI ranks.
   *
   * Workflow:
   * - Allocates velocity and force arrays (each of shape [num_particles, 3]).
   * - Initializes the velocity array to zero.
   * - Copies the input force array into device memory.
   * - Reorders the force array aligning with the node ordering of the
   * clustering tree.
   * - Calls DenseDot to compute velocities.
   * - Collects the result across MPI ranks using all-reduce.
   * - Restores the velocity array to the original ordering.
   * - Copies the result back into the output Python array.
   *
   * @param uArray [out] The computed velocities, numpy array of shape
   * (num_particles, 3).
   * @param fArray [in]  The input forces, numpy array of shape (num_particles,
   * 3).
   */
  void DenseDot(pybind11::array_t<float> &uArray,
                pybind11::array_t<float> &fArray);

  /**
   * @brief Updates the particle coordinates and triggers an internal update of
   * the clustering tree.
   *
   * It updates the internally stored particle coordinates
   * `mCoordPtr` with the values from the input `coord`. After the update,
   * it calls the `Update()` function to perform the rebuilding of the
   * clustering tree w.r.t. the new particle coordinates.
   *
   * @param coord [in] The array representing the new coordinates for the
   * particles. Size (num_particles, nDim). If nDim = 3, it stands for (x, y,
   * z). If nDim = 6, it stands for (x, y, z, rx, ry, rz) which is not
   * implemented yet.
   */
  void UpdateCoord(pybind11::array_t<float> &coord);

  /**
   * @brief Sets the value of epsilon used by the adaptive cross approximation
   *
   * It sets the value of the internal `mEpsilon` variable, which
   * controls the convergence criteria of the adaptive cross approximation used
   * in function FarDot.
   *
   * @param epsilon [in] The value to set for epsilon.
   */
  void SetEpsilon(const double epsilon);

  /**
   * @brief Set the eta parameter used in the clustering tree.
   *
   * This updates the mEta variable, affecting how close/far pairs are
   * determined in the clustering tree.
   *
   * @param eta [in] The value to set for eta.
   */
  void SetEta(const double eta);

  /**
   * @brief Set the maximum number of iterations for the adaptive cross
   * approximation.
   *
   * This updates mMaxIter, controlling how many iterations are performed by the
   * algorithm.
   *
   * @param maxIter [in] The maximum number of iterations to perform.
   */
  void SetMaxIter(const int maxIter);

  /**
   * @brief Sets the factor that determines the size of the matrix pool.
   *
   * FarDot requires a spare storage space allocated in front to avoid the
   * frequent allocation and deallocation of memory on GPU used for storing C
   * and Q used by the adaptive cross approximation.
   *
   * @param factor [in] The factor to adjust the matrix pool size.
   */
  void SetMatPoolSizeFactor(const int factor);

  /**
   * @brief Sets the flag to enable or disable the post-check operation.
   *
   * @param flag [in] A boolean flag indicating whether the post-check operation
   * is enabled (true) or disabled (false).
   */
  void SetPostCheckFlag(const bool flag);

  /**
   * @brief Sets the flag to enable or disable the use of symmetry in the model.
   *
   * The mobility tensor has a symmetric pattern by definition. Enabling this
   * flag can save the number of queries when doing CloseDot and FarDot.
   *
   * @param flag [in] A boolean flag indicating whether symmetry is enabled
   * (true) or disabled (false).
   */
  void SetUseSymmetryFlag(const bool flag);

  /**
   * @brief Sets the maximum number of node pairs for far-range interaction
   * computations in function FarDot.
   *
   * It sets the limit on the number of node pairs that can
   * simultaneously involved in far-range interaction computations. This value
   * helps in managing computational resources.
   *
   * @param size [in] The maximum number of node pairs to be considered for
   * far-range interaction computations.
   */
  void SetMaxFarDotWorkNodeSize(const int size);

  /**
   * @brief Sets the maximum number of relative coordinates can be
   * simultaneously processed.
   *
   * It sets the upper limit on the number of relative coordinates
   * that can be handled by the function CloseDot and FarDot. It also determines
   * the maximum number of queries can be performed by the two-body interaction
   * model when evaluating the entries in the mobility tensor.
   *
   * @param size [in] The maximum number of relative coordinates that can be
   * processed.
   */
  void SetMaxRelativeCoord(const size_t size);

  /**
   * @brief Set the far-field cut-off distance for interactions (not used
   * anymore).
   *
   * Updates mMaxFarFieldDistance, the cut-off for far-range interactions.
   *
   * @param distance [in] The cut-off distance.
   */
  void SetMaxFarFieldDistance(const double distance);

  /**
   * @brief Reorders the rows of the given device matrix v based on the provided
   * index mapping.
   *
   * Produces a reordered copy of the matrix v so that row i of the result is
   * taken from row reorderedMap[i] of the original matrix. The matrix v is
   * updated in place in device memory.
   *
   * @param reorderedMap A vector of size (num_particles) specifying the new
   * order for the rows of v.
   * @param v The device matrix of size (num_particles, 3) to be reordered in
   * place.
   */
  void Reorder(const std::vector<std::size_t> &reorderedMap,
               DeviceDoubleMatrix v);

  /**
   * @brief Reverses the reordering of the rows of the given matrix v based on
   * the provided index mapping.
   *
   * It restores the original row order of the matrix v (of size
   * [num_particles, 3]) using the mapping provided by reorderedMap (vector of
   * size num_nodes). The mapping is applied on the device using Kokkos
   * parallelism for efficient performance.
   *
   * @param reorderedMap A vector of size (num_particles) specifying the
   * original row order to restore in v.
   * @param v The device matrix of size (num_particles, 3) to be reordered back.
   * After the function, v will have its rows placed back into the original
   * order.
   */
  void BackwardReorder(const std::vector<std::size_t> &reorderedMap,
                       DeviceDoubleMatrix v);
};

/** @} */

#endif