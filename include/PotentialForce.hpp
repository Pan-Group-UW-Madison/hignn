#ifndef _PotentialForce_Hpp_
#define _PotentialForce_Hpp_

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <vector>
#include <chrono>

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <mpi.h>

#include "Typedef.hpp"
#include "PointCloudSearch.hpp"

class __attribute__((visibility("default"))) PotentialForce {
private:
  HostFloatMatrix mSourceSites;
  HostFloatMatrix mTargetSites;
  HostFloatMatrix mPotentialForce;
  HostFloatVector mEpsilonLists;
  HostIndexVector mSourceIndex;
  HostIndexMatrix mTwoBodyEdgeInfo;
  HostIndexMatrix mTargetNeighborLists;

  HostIndexVector mTwoBodyEdgeNum;
  HostIndexVector mTwoBodyEdgeOffset;

  bool mIsPeriodicBoundary;
  float mTwoBodyEpsilon, mThreeBodyEpsilon;
  float mDomainLow[3], mDomainHigh[3];
  int mDim;

  std::shared_ptr<PointCloudSearch<HostFloatMatrix>> mPointCloudSearch;

public:
  PotentialForce() : mIsPeriodicBoundary(false), mTwoBodyEpsilon(0.0), mDim(3) {
  }

  ~PotentialForce() {
  }

  inline void CalculatePotentialForce(float *r, float *f) {
    const float De = 1.0;
    const float a = 1.0;
    const float re = 2.5;

    float rNorm = 0.0;
    float fMag = 0.0;

    rNorm = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    fMag =
        -2.0 * a * De * (exp(-a * (rNorm - re)) - exp(-2.0 * a * (rNorm - re)));

    float ratio = fMag / rNorm;
    for (int j = 0; j < 3; j++) {
      f[j] = r[j] * ratio;
    }
  }

  void SetPeriodic(bool isPeriodicBoundary = false) {
    mIsPeriodicBoundary = isPeriodicBoundary;
  }

  void SetDomain(pybind11::array_t<float> domain) {
    pybind11::buffer_info buf = domain.request();

    if (buf.ndim != 2) {
      throw std::runtime_error("Rank of domain must be two");
    }

    auto data = domain.unchecked<2>();

    mDomainLow[0] = data(0, 0);
    mDomainLow[1] = data(0, 1);
    mDomainLow[2] = data(0, 2);

    mDomainHigh[0] = data(1, 0);
    mDomainHigh[1] = data(1, 1);
    mDomainHigh[2] = data(1, 2);

    mIsPeriodicBoundary = true;
  }

  void UpdateCoord(pybind11::array_t<float> coord) {
    pybind11::buffer_info buf = coord.request();

    if (buf.ndim != 2) {
      throw std::runtime_error("Rank of target sites must be two");
    }

    mTargetSites =
        decltype(mTargetSites)("target sites", (std::size_t)coord.shape(0),
                               (std::size_t)coord.shape(1));

    auto data = coord.unchecked<2>();

    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
                             0, mTargetSites.extent(0)),
                         [=](size_t i) {
                           mTargetSites(i, 0) = data(i, 0);
                           mTargetSites(i, 1) = data(i, 1);
                           mTargetSites(i, 2) = data(i, 2);
                         });
    Kokkos::fence();

    mTargetNeighborLists = decltype(mTargetNeighborLists)(
        "two body neighbor lists", mTargetSites.extent(0), 2);
    mEpsilonLists =
        decltype(mEpsilonLists)("epsilon lists", mTargetSites.extent(0));
  }

  void SetTwoBodyEpsilon(float epsilon) {
    mTwoBodyEpsilon = epsilon;
  }

  void BuildSourceSites() {
    if (!mIsPeriodicBoundary) {
      mSourceSites = mTargetSites;

      mSourceIndex =
          decltype(mSourceIndex)("source index", mSourceSites.extent(0));
      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
              0, mSourceIndex.extent(0)),
          [&](size_t i) { mSourceIndex(i) = i; });
      Kokkos::fence();
    } else {
      float epsilon = mTwoBodyEpsilon;

      float coreDomainLow[3], coreDomainHigh[3], domainSize[3];

      for (int i = 0; i < 3; i++)
        coreDomainLow[i] = mDomainLow[i] + epsilon;
      for (int i = 0; i < 3; i++)
        coreDomainHigh[i] = mDomainHigh[i] - epsilon;
      for (int i = 0; i < 3; i++)
        domainSize[i] = mDomainHigh[i] - mDomainLow[i];

      const std::size_t numTarget = mTargetSites.extent(0);

      HostIndexVector numSourceDuplicate =
          HostIndexVector("num source duplicate", numTarget);
      HostIntMatrix axisSourceDuplicate =
          HostIntMatrix("axis source duplicate", numTarget, 3);

      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numTarget),
          [&](size_t i) {
            std::size_t num = 1;
            for (int j = 0; j < 3; j++) {
              axisSourceDuplicate(i, j) = 1;
              if (mTargetSites(i, j) < coreDomainLow[j]) {
                axisSourceDuplicate(i, j) = -2;
                num *= 2;
              } else if (mTargetSites(i, j) > coreDomainHigh[j]) {
                axisSourceDuplicate(i, j) = 2;
                num *= 2;
              } else {
                axisSourceDuplicate(i, j) = 1;
              }
            }

            numSourceDuplicate(i) = num;
          });
      Kokkos::fence();

      HostIndexVector numSourceOffset =
          HostIndexVector("num source offset", numTarget + 1);

      numSourceOffset(0) = 0;
      for (size_t i = 0; i < numTarget; i++) {
        numSourceOffset(i + 1) = numSourceOffset(i) + numSourceDuplicate(i);
      }
      std::size_t numSource = numSourceOffset(numTarget);

      mSourceSites = decltype(mSourceSites)("source sites", numSource, 3);
      mSourceIndex = decltype(mSourceIndex)("source index", numSource);

      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numTarget),
          [&](size_t i) {
            std::vector<float> offset;
            offset.resize(3 * numSourceDuplicate(i));

            const std::size_t num = numSourceDuplicate(i);
            std::size_t stride1 = numSourceDuplicate(i);
            for (int j = 0; j < 3; j++) {
              if (axisSourceDuplicate(i, j) == 1) {
                for (size_t n = 0; n < num; n++) {
                  offset[n * 3 + j] = 0;
                }
              }
              if (axisSourceDuplicate(i, j) == 2) {
                for (size_t m = 0; m < num; m += stride1) {
                  for (size_t n = m; n < m + stride1 / 2; n++) {
                    offset[n * 3 + j] = 0;
                  }
                  for (size_t n = m + stride1 / 2; n < m + stride1; n++) {
                    offset[n * 3 + j] = -domainSize[j];
                  }
                }
                stride1 /= 2;
              }
              if (axisSourceDuplicate(i, j) == -2) {
                for (size_t m = 0; m < num; m += stride1) {
                  for (size_t n = m; n < m + stride1 / 2; n++) {
                    offset[n * 3 + j] = 0;
                  }
                  for (size_t n = m + stride1 / 2; n < m + stride1; n++) {
                    offset[n * 3 + j] = domainSize[j];
                  }
                }
                stride1 /= 2;
              }
            }

            for (size_t m = numSourceOffset[i]; m < numSourceOffset[i + 1];
                 m++) {
              for (int j = 0; j < 3; j++) {
                mSourceSites(m, j) = mTargetSites(i, j) +
                                     offset[(m - numSourceOffset[i]) * 3 + j];
              }
              mSourceIndex(m) = i;
            }
          });
      Kokkos::fence();
    }

    mPointCloudSearch = std::make_shared<PointCloudSearch<HostFloatMatrix>>(
        CreatePointCloudSearch(mSourceSites, 3));
  }

  void BuildTargetNeighborLists(float epsilon) {
    BuildSourceSites();

    auto numTarget = mTargetSites.extent(0);

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numTarget),
        [&](size_t i) { mEpsilonLists(i) = epsilon; });
    Kokkos::fence();

    auto numNeighbor =
        1 + mPointCloudSearch->generate2DNeighborListsFromRadiusSearch(
                true, mTargetSites, mTargetNeighborLists, mEpsilonLists, 0.0,
                epsilon);
    if (numNeighbor > mTargetNeighborLists.extent(1))
      Kokkos::resize(mTargetNeighborLists, numTarget, numNeighbor);

    mPointCloudSearch->generate2DNeighborListsFromRadiusSearch(
        false, mTargetSites, mTargetNeighborLists, mEpsilonLists, 0.0, epsilon);

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numTarget),
        [&](size_t i) {
          // change the target index to the real index
          size_t counter = 0;
          while (counter < mTargetNeighborLists(i, 0)) {
            mTargetNeighborLists(i, counter + 1) =
                mSourceIndex(mTargetNeighborLists(i, counter + 1));
            counter++;
          }
        });
    Kokkos::fence();

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numTarget),
        [&](size_t i) {
          // ensure the current target index appears at the
          // beginning of the list
          size_t counter = 0;
          while (counter < mTargetNeighborLists(i, 0)) {
            if (mTargetNeighborLists(i, counter + 1) == i) {
              std::swap(mTargetNeighborLists(i, 1),
                        mTargetNeighborLists(i, counter + 1));
              break;
            }
            counter++;
          }
        });
    Kokkos::fence();
  }

  pybind11::array_t<float> GetPotentialForce(pybind11::array_t<float> coord) {
    int mpiRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

    UpdateCoord(coord);

    if (mpiRank == 0) {
      std::cout << "  Potential force updated coord" << std::endl;
    }

    BuildTargetNeighborLists(mTwoBodyEpsilon);

    if (mpiRank == 0) {
      std::cout << "  Potential force built target neighbor lists" << std::endl;
    }

    std::size_t numTarget = mTargetSites.extent(0);

    Kokkos::resize(mTwoBodyEdgeNum, numTarget);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numTarget),
        [=](std::size_t i) {
          mTwoBodyEdgeNum(i) = mTargetNeighborLists(i, 0) - 1;
        });
    Kokkos::fence();

    Kokkos::resize(mTwoBodyEdgeOffset, numTarget);

    Kokkos::parallel_scan(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numTarget),
        [&](const int i, size_t &sum, [[maybe_unused]] bool final) {
          mTwoBodyEdgeOffset(i) = sum;
          sum += mTwoBodyEdgeNum(i);
        });
    Kokkos::fence();

    const int numEdge =
        mTwoBodyEdgeOffset(numTarget - 1) + mTwoBodyEdgeNum(numTarget - 1);

    mTwoBodyEdgeInfo =
        decltype(mTwoBodyEdgeInfo)("two body edge info", numEdge, 2);
    mPotentialForce =
        decltype(mPotentialForce)("potential force", numTarget, 3);

    if (mpiRank == 0) {
      std::cout << "  Start of calculating potential force" << std::endl;
    }

    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numTarget),
        [&](size_t i) {
          std::size_t offset = 0;
          for (int j = 0; j < 3; j++)
            mPotentialForce(i, j) = 0.0;

          for (std::size_t j = 1; j < mTargetNeighborLists(i, 0); j++) {
            std::size_t neighborIdx = mTargetNeighborLists(i, j + 1);
            mTwoBodyEdgeInfo(mTwoBodyEdgeOffset(i) + offset, 0) = i;
            mTwoBodyEdgeInfo(mTwoBodyEdgeOffset(i) + offset, 1) =
                mSourceIndex(neighborIdx);

            float r[3], f[3];
            for (int k = 0; k < 3; k++)
              r[k] = mTargetSites(i, k) - mSourceSites(neighborIdx, k);

            float rNorm = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);

            if (rNorm > 1e-3) {
              CalculatePotentialForce(r, f);
              for (int k = 0; k < 3; k++)
                mPotentialForce(i, k) += f[k];
            }

            offset++;
          }
        });
    Kokkos::fence();

    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    if (mpiRank == 0)
      printf("  End of calculating potential force. Time: %.4fs\n",
             (double)duration / 1e6);

    pybind11::array_t<float> potentialForce(
        {mPotentialForce.extent(0), mPotentialForce.extent(1)});

    pybind11::buffer_info buf = potentialForce.request();
    auto ptr = (float *)buf.ptr;

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numTarget),
        [=](size_t i) {
          for (int j = 0; j < 3; j++)
            ptr[i * 3 + j] = mPotentialForce(i, j);
        });
    Kokkos::fence();

    return potentialForce;
  }
};

#endif