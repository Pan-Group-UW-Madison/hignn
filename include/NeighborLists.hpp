#ifndef _NeighborLists_Hpp_
#define _NeighborLists_Hpp_

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

class __attribute__((visibility("default"))) NeighborLists {
private:
  HostFloatMatrix mSourceSites;
  HostFloatMatrix mTargetSites;
  HostIndexVector mSourceIndex;

  HostIndexMatrix mTargetNeighborLists;
  HostFloatVector mEpsilonLists;

  HostIndexMatrix mTwoBodyEdgeInfo;
  HostIndexVector mTwoBodyEdgeNum;
  HostIndexVector mTwoBodyEdgeOffset;

  HostIndexMatrix mThreeBodyEdgeInfo;
  HostIndexMatrix mThreeBodyEdgeSelfInfo;
  HostIndexVector mThreeBodyEdgeOffset;
  HostIndexVector mThreeBodyEdgeSelfOffset;
  HostIndexVector mThreeBodyEdgeNum;
  HostIndexVector mThreeBodyEdgeSelfNum;

  bool mIsPeriodicBoundary;
  float mTwoBodyEpsilon, mThreeBodyEpsilon;
  float mDomainLow[3], mDomainHigh[3];
  int mDim;

  std::shared_ptr<PointCloudSearch<HostFloatMatrix>> mPointCloudSearch;

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

public:
  NeighborLists()
      : mIsPeriodicBoundary(false),
        mTwoBodyEpsilon(0.0),
        mThreeBodyEpsilon(0.0),
        mDim(3) {
  }

  ~NeighborLists() {
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
                         [&](size_t i) {
                           mTargetSites(i, 0) = data(i, 0);
                           mTargetSites(i, 1) = data(i, 1);
                           mTargetSites(i, 2) = data(i, 2);
                         });
    Kokkos::fence();

    mTargetNeighborLists = decltype(mTargetNeighborLists)(
        "neighbor lists", mTargetSites.extent(0), 2);

    mEpsilonLists =
        decltype(mEpsilonLists)("epsilon lists", mTargetSites.extent(0));
  }

  void SetTwoBodyEpsilon(float epsilon) {
    mTwoBodyEpsilon = epsilon;
  }

  void SetThreeBodyEpsilon(float epsilon) {
    mThreeBodyEpsilon = epsilon;
  }

  pybind11::array_t<std::size_t> GetTwoBodyEdgeInfo() {
    pybind11::array_t<std::size_t> edgeInfo(
        {mTwoBodyEdgeInfo.extent(1), mTwoBodyEdgeInfo.extent(0)});

    pybind11::buffer_info buf = edgeInfo.request();
    auto ptr = (size_t *)buf.ptr;

    const int numEdge = mTwoBodyEdgeInfo.extent(0);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numEdge),
        [&](size_t i) { ptr[i] = mTwoBodyEdgeInfo(i, 0); });
    Kokkos::fence();
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numEdge),
        [&](size_t i) { ptr[numEdge + i] = mTwoBodyEdgeInfo(i, 1); });
    Kokkos::fence();

    return edgeInfo;
  }

  pybind11::array_t<std::size_t> GetTwoBodyEdgeInfoByIndex(
      pybind11::array_t<std::size_t> targetIndex) {
    auto data = targetIndex.unchecked<1>();

    std::size_t numTarget = (std::size_t)targetIndex.shape(0);

    std::size_t totalTarget = mTargetSites.extent(0);
    std::size_t edgePerTarget = mTargetSites.extent(0) - 1;
    std::size_t numEdge = numTarget * edgePerTarget;

    pybind11::array_t<std::size_t> edgeInfo(
        {mTwoBodyEdgeInfo.extent(1), numEdge});

    pybind11::buffer_info buf = edgeInfo.request();
    auto ptr = (size_t *)buf.ptr;

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numTarget),
        [&](size_t i) {
          unsigned int index = data(i);
          std::size_t counter = i * edgePerTarget;
          for (unsigned int j = 0; j < totalTarget; j++) {
            if (j != index) {
              ptr[counter] = j;
              ptr[numEdge + counter] = index;
              counter++;
            }
          }
        });

    return edgeInfo;
  }

  pybind11::array_t<std::size_t> GetThreeBodyEdgeInfo() {
    pybind11::array_t<std::size_t> edgeInfo(
        {mThreeBodyEdgeInfo.extent(1), mThreeBodyEdgeInfo.extent(0)});

    pybind11::buffer_info buf = edgeInfo.request();
    auto ptr = (size_t *)buf.ptr;

    const int numEdge = mThreeBodyEdgeInfo.extent(0);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numEdge),
        [&](size_t i) {
          ptr[i] = mThreeBodyEdgeInfo(i, 0);
          ptr[numEdge + i] = mThreeBodyEdgeInfo(i, 1);
          ptr[2 * numEdge + i] = mThreeBodyEdgeInfo(i, 2);
        });
    Kokkos::fence();

    return edgeInfo;
  }

  pybind11::array_t<std::size_t> GetThreeBodyEdgeInfoByIndex(
      pybind11::array_t<std::size_t> targetIndex) {
    auto data = targetIndex.unchecked<1>();

    std::size_t numTarget = (std::size_t)targetIndex.shape(0);

    // count edge and calculate offset
    Kokkos::View<size_t *, Kokkos::DefaultHostExecutionSpace> edgeLocalOffset(
        "edge local offset", numTarget);
    Kokkos::parallel_scan(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numTarget),
        [&](const int i, size_t &lSum, [[maybe_unused]] bool final) {
          edgeLocalOffset(i) = lSum;
          lSum += mThreeBodyEdgeNum(data(i));
        });
    Kokkos::fence();

    std::size_t numEdge =
        edgeLocalOffset(numTarget - 1) + mThreeBodyEdgeNum(data(numTarget - 1));

    pybind11::array_t<std::size_t> edgeInfo(
        {mThreeBodyEdgeInfo.extent(1), numEdge});

    pybind11::buffer_info buf = edgeInfo.request();
    auto ptr = (size_t *)buf.ptr;

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numTarget),
        [&](size_t i) {
          unsigned int index = data(i);
          unsigned int offset = mThreeBodyEdgeOffset(index);
          unsigned int numIndexEdge = mThreeBodyEdgeNum(index);
          for (unsigned int j = 0; j < numIndexEdge; j++) {
            std::size_t counter = edgeLocalOffset(i) + j;
            ptr[counter] = mThreeBodyEdgeInfo(offset + j, 2);
            ptr[numEdge + counter] = mThreeBodyEdgeInfo(offset + j, 1);
            ptr[2 * numEdge + counter] = mThreeBodyEdgeInfo(offset + j, 0);
          }
        });

    return edgeInfo;
  }

  pybind11::array_t<std::size_t> GetThreeBodyEdgeSelfInfo() {
    pybind11::array_t<std::size_t> edgeInfo(
        {mThreeBodyEdgeSelfInfo.extent(1), mThreeBodyEdgeSelfInfo.extent(0)});

    pybind11::buffer_info buf = edgeInfo.request();
    auto ptr = (size_t *)buf.ptr;

    const int numEdge = mThreeBodyEdgeSelfInfo.extent(0);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numEdge),
        [&](size_t i) {
          ptr[i] = mThreeBodyEdgeSelfInfo(i, 0);
          ptr[numEdge + i] = mThreeBodyEdgeSelfInfo(i, 1);
        });
    Kokkos::fence();

    return edgeInfo;
  }

  pybind11::array_t<std::size_t> GetThreeBodyEdgeSelfInfoByIndex(
      pybind11::array_t<std::size_t> targetIndex) {
    auto data = targetIndex.unchecked<1>();

    std::size_t numTarget = (std::size_t)targetIndex.shape(0);

    // count edge and calculate offset
    Kokkos::View<size_t *, Kokkos::DefaultHostExecutionSpace>
        edgeSelfLocalOffset("edge self local offset", numTarget);
    Kokkos::parallel_scan(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numTarget),
        [&](const int i, size_t &lSum, [[maybe_unused]] bool final) {
          edgeSelfLocalOffset(i) = lSum;
          lSum += mThreeBodyEdgeSelfNum(data(i));
        });
    Kokkos::fence();

    std::size_t numEdge = edgeSelfLocalOffset(numTarget - 1) +
                          mThreeBodyEdgeSelfNum(data(numTarget - 1));

    pybind11::array_t<std::size_t> edgeSelfInfo(
        {mThreeBodyEdgeSelfInfo.extent(1), numEdge});

    pybind11::buffer_info buf = edgeSelfInfo.request();
    auto ptr = (size_t *)buf.ptr;

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numTarget),
        [&](size_t i) {
          unsigned int index = data(i);
          unsigned int offset = mThreeBodyEdgeSelfOffset(index);
          unsigned int numIndexEdge = mThreeBodyEdgeSelfNum(index);
          for (unsigned int j = 0; j < numIndexEdge; j++) {
            std::size_t counter = edgeSelfLocalOffset(i) + j;
            ptr[counter] = mThreeBodyEdgeSelfInfo(offset + j, 1);
            ptr[numEdge + counter] = mThreeBodyEdgeSelfInfo(offset + j, 0);
          }
        });

    return edgeSelfInfo;
  }

  void BuildTwoBodyInfo() {
    BuildTargetNeighborLists(mTwoBodyEpsilon);

    std::size_t numTarget = mTargetSites.extent(0);

    Kokkos::resize(mTwoBodyEdgeNum, numTarget);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numTarget),
        [&](std::size_t i) {
          mTwoBodyEdgeNum(i) = mTargetNeighborLists(i, 0) - 1;
        });
    Kokkos::fence();

    Kokkos::resize(mTwoBodyEdgeOffset, numTarget);

    Kokkos::parallel_scan(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numTarget),
        [&](const int i, size_t &lSum, [[maybe_unused]] bool final) {
          mTwoBodyEdgeOffset(i) = lSum;
          lSum += mTwoBodyEdgeNum(i);
        });
    Kokkos::fence();

    const int numEdge =
        mTwoBodyEdgeOffset(numTarget - 1) + mTwoBodyEdgeNum(numTarget - 1);

    mTwoBodyEdgeInfo =
        decltype(mTwoBodyEdgeInfo)("two body edge info", numEdge, 2);

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numTarget),
        [&](size_t i) {
          std::size_t offset = 0;

          for (std::size_t j = 1; j < mTargetNeighborLists(i, 0); j++) {
            std::size_t neighborIdx = mTargetNeighborLists(i, j + 1);
            mTwoBodyEdgeInfo(mTwoBodyEdgeOffset(i) + offset, 0) = i;
            mTwoBodyEdgeInfo(mTwoBodyEdgeOffset(i) + offset, 1) =
                mSourceIndex(neighborIdx);

            offset++;
          }
        });
    Kokkos::fence();
  }

  void BuildThreeBodyInfo() {
    BuildTargetNeighborLists(mThreeBodyEpsilon);

    std::size_t numTarget = mTargetSites.extent(0);

    Kokkos::resize(mThreeBodyEdgeNum, numTarget);
    Kokkos::resize(mThreeBodyEdgeSelfNum, numTarget);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numTarget),
        [&](std::size_t i) {
          mThreeBodyEdgeNum(i) = 0;
          mThreeBodyEdgeSelfNum(i) = mTargetNeighborLists(i, 0) - 1;
          for (std::size_t j = 1; j < mTargetNeighborLists(i, 0); j++) {
            int neighborIdx = mTargetNeighborLists(i, j + 1);
            mThreeBodyEdgeNum(i) += (mTargetNeighborLists(neighborIdx, 0) - 2);
          }
        });
    Kokkos::fence();

    Kokkos::resize(mThreeBodyEdgeOffset, numTarget);
    Kokkos::resize(mThreeBodyEdgeSelfOffset, numTarget);

    Kokkos::parallel_scan(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numTarget),
        [&](const int i, size_t &lSum, [[maybe_unused]] bool final) {
          mThreeBodyEdgeOffset(i) = lSum;
          lSum += mThreeBodyEdgeNum(i);
        });
    Kokkos::fence();
    Kokkos::parallel_scan(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numTarget),
        [&](const int i, size_t &lSum, [[maybe_unused]] bool final) {
          mThreeBodyEdgeSelfOffset(i) = lSum;
          lSum += mThreeBodyEdgeSelfNum(i);
        });
    Kokkos::fence();

    Kokkos::resize(
        mThreeBodyEdgeInfo,
        mThreeBodyEdgeOffset(numTarget - 1) + mThreeBodyEdgeNum(numTarget - 1),
        3);
    Kokkos::resize(mThreeBodyEdgeSelfInfo,
                   mThreeBodyEdgeSelfOffset(numTarget - 1) +
                       mThreeBodyEdgeSelfNum(numTarget - 1),
                   2);

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numTarget),
        [&](size_t i) {
          std::size_t offset = 0;
          for (std::size_t j = 1; j < mTargetNeighborLists(i, 0); j++) {
            std::size_t neighborIdx = mTargetNeighborLists(i, j + 1);
            for (std::size_t it = 1; it < mTargetNeighborLists(neighborIdx, 0);
                 it++) {
              std::size_t neighbor_neighborIdx =
                  mTargetNeighborLists(neighborIdx, it + 1);
              if (neighbor_neighborIdx != i) {
                mThreeBodyEdgeInfo(mThreeBodyEdgeOffset(i) + offset, 0) = i;
                mThreeBodyEdgeInfo(mThreeBodyEdgeOffset(i) + offset, 1) =
                    neighborIdx;
                mThreeBodyEdgeInfo(mThreeBodyEdgeOffset(i) + offset, 2) =
                    neighbor_neighborIdx;

                offset++;
              }
            }

            mThreeBodyEdgeSelfInfo(mThreeBodyEdgeSelfOffset(i) + j - 1, 0) = i;
            mThreeBodyEdgeSelfInfo(mThreeBodyEdgeSelfOffset(i) + j - 1, 1) =
                neighborIdx;
          }
        });
    Kokkos::fence();
  }
};

#endif
