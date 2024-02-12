#ifndef _ThreeBodyEdge_hpp_
#define _ThreeBodyEdge_hpp_

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <vector>

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Typedef.hpp"
#include "PointCloudSearch.hpp"

// TODO: mMorseForce and three_body_edge_attr if it is selected by index, it is
// still problematic

class neighbor_lists {
private:
  HostFloatMatrix mSourceSites;
  HostFloatMatrix mTargetSites;
  HostFloatMatrix mMorseForce;
  HostFloatVector mEpsilonLists;
  HostIndexVector mSourceIndex;
  HostIndexMatrix mTwoBodyEdgeInfo;
  HostIndexMatrix mThreeBodyEdgeInfo;
  HostIndexMatrix mThreeBodyEdgeSelfInfo;
  HostIndexMatrix mTargetNeighborLists;

  HostIndexVector mTwoBodyEdgeNum;
  HostIndexVector mThreeBodyEdgeNum;
  HostIndexVector mThreeBodyEdgeSelfNum;
  HostIndexVector mTwoBodyEdgeOffset;
  HostIndexVector mThreeBodyEdgeOffset;
  HostIndexVector mThreeBodyEdgeSelfOffset;

  bool mIsSourceSitesBuilt;
  bool mIsPeriodicBoundary;
  float mTwoBodyEpsilon, mThreeBodyEpsilon;
  float mDomainLow[3], mDomainHigh[3];
  int mDim;

  std::shared_ptr<PointCloudSearch<HostFloatMatrix>> mPointCloudSearch;

public:
  neighbor_lists()
      : mIsSourceSitesBuilt(false),
        mIsPeriodicBoundary(false),
        mDim(3),
        mTwoBodyEpsilon(0.0),
        mThreeBodyEpsilon(0.0) {
  }

  ~neighbor_lists() {
  }

  inline void calculate_morse(float *r, float *f) {
    const float De = 1.0;
    const float a = 1.0;
    const float re = 2.5;

    float r_norm = 0.0;
    float F_mag = 0.0;

    r_norm = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    F_mag = -2.0 * a * De *
            (exp(-a * (r_norm - re)) - exp(-2.0 * a * (r_norm - re)));

    float ratio = F_mag / r_norm;
    for (int j = 0; j < 3; j++) {
      f[j] = r[j] * ratio;
    }
  }

  void set_periodic(bool _is_periodic_boundary = false) {
    mIsPeriodicBoundary = _is_periodic_boundary;
  }

  void set_domain(pybind11::array_t<float> _domain) {
    pybind11::buffer_info buf = _domain.request();

    if (buf.ndim != 2) {
      throw std::runtime_error("Rank of domain must be two");
    }

    auto data = _domain.unchecked<2>();

    mDomainLow[0] = data(0, 0);
    mDomainLow[1] = data(0, 1);
    mDomainLow[2] = data(0, 2);

    mDomainHigh[0] = data(1, 0);
    mDomainHigh[1] = data(1, 1);
    mDomainHigh[2] = data(1, 2);
  }

  void set_dimension(int _dim) {
    mDim = _dim;
  }

  void set_target_sites(pybind11::array_t<float> _target_sites) {
    pybind11::buffer_info buf = _target_sites.request();

    if (buf.ndim != 2) {
      throw std::runtime_error("Rank of target sites must be two");
    }

    mTargetSites = decltype(mTargetSites)("target sites",
                                          (std::size_t)_target_sites.shape(0),
                                          (std::size_t)_target_sites.shape(1));

    auto data = _target_sites.unchecked<2>();

    Kokkos::parallel_for(
        Kokkos::RangePolicy<host_execution_space>(0, mTargetSites.extent(0)),
        [=](size_t i) {
          mTargetSites(i, 0) = data(i, 0);
          mTargetSites(i, 1) = data(i, 1);
          mTargetSites(i, 2) = data(i, 2);
        });
    Kokkos::fence();

    mTargetNeighborLists = decltype(mTargetNeighborLists)(
        "two body neighbor lists", mTargetSites.extent(0), 2);
    mTargetNeighborLists = decltype(mTargetNeighborLists)(
        "three body neighbor lists", mTargetSites.extent(0), 2);
    mEpsilonLists =
        decltype(mEpsilonLists)("epsilon lists", mTargetSites.extent(0));

    mIsSourceSitesBuilt = false;
  }

  void set_two_body_epsilon(float _epsilon) {
    mTwoBodyEpsilon = _epsilon;
  }

  void set_three_body_epsilon(float _epsilon) {
    mThreeBodyEpsilon = _epsilon;
  }

  pybind11::array_t<std::size_t> get_two_body_edge_info() {
    pybind11::array_t<std::size_t> py_edge_info(
        {mTwoBodyEdgeInfo.extent(1), mTwoBodyEdgeInfo.extent(0)});

    pybind11::buffer_info buf = py_edge_info.request();
    auto ptr = (size_t *)buf.ptr;

    const int num_edge = mTwoBodyEdgeInfo.extent(0);
    Kokkos::parallel_for(Kokkos::RangePolicy<host_execution_space>(0, num_edge),
                         [=](size_t i) { ptr[i] = mTwoBodyEdgeInfo(i, 0); });
    Kokkos::fence();
    Kokkos::parallel_for(
        Kokkos::RangePolicy<host_execution_space>(0, num_edge),
        [=](size_t i) { ptr[num_edge + i] = mTwoBodyEdgeInfo(i, 1); });
    Kokkos::fence();

    return py_edge_info;
  }

  pybind11::array_t<std::size_t> get_two_body_edge_info_by_index(
      pybind11::array_t<std::size_t> _target_index) {
    auto data = _target_index.unchecked<1>();

    std::size_t num_target = (std::size_t)_target_index.shape(0);

    std::size_t totalTarget = mTargetSites.extent(0);
    std::size_t edgePerTarget = mTargetSites.extent(0) - 1;
    std::size_t numEdge = num_target * edgePerTarget;

    pybind11::array_t<std::size_t> py_edge_info(
        {mTwoBodyEdgeInfo.extent(1), numEdge});

    pybind11::buffer_info buf = py_edge_info.request();
    auto ptr = (size_t *)buf.ptr;

    Kokkos::parallel_for(
        Kokkos::RangePolicy<host_execution_space>(0, num_target),
        [=](size_t i) {
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

    // // count edge and calculate offset
    // Kokkos::View<size_t *, host_execution_space> edge_local_offset(
    //     "edge local offset", num_target);
    // Kokkos::parallel_scan(
    //     Kokkos::RangePolicy<host_execution_space>(0, num_target),
    //     KOKKOS_LAMBDA(const int i, size_t &lsum, bool final) {
    //       edge_local_offset(i) = lsum;
    //       lsum += mTwoBodyEdgeNum(data(i));
    //     });
    // Kokkos::fence();

    // std::size_t numEdge = edge_local_offset(num_target - 1) +
    //                       mTwoBodyEdgeNum(data(num_target - 1));

    // pybind11::array_t<std::size_t> py_edge_info(
    //     {mTwoBodyEdgeInfo.extent(1), numEdge});

    // pybind11::buffer_info buf = py_edge_info.request();
    // auto ptr = (size_t *)buf.ptr;

    // Kokkos::parallel_for(
    //     Kokkos::RangePolicy<host_execution_space>(0, num_target),
    //     [=](size_t i) {
    //       unsigned int index = data(i);
    //       unsigned int offset = mTwoBodyEdgeOffset(index);
    //       unsigned int num_index_edge = mTwoBodyEdgeNum(index);
    //       for (unsigned int j = 0; j < num_index_edge; j++) {
    //         std::size_t counter = edge_local_offset(i) + j;
    //         ptr[counter] = mTwoBodyEdgeInfo(offset + j, 0);
    //         ptr[numEdge + counter] = mTwoBodyEdgeInfo(offset + j, 1);
    //       }
    //     });

    return py_edge_info;
  }

  pybind11::array_t<float> get_morse_force() {
    pybind11::array_t<float> py_morse_force(
        {mMorseForce.extent(0), mMorseForce.extent(1)});

    pybind11::buffer_info buf = py_morse_force.request();
    auto ptr = (float *)buf.ptr;

    const int num_edge = mMorseForce.extent(0);
    Kokkos::parallel_for(Kokkos::RangePolicy<host_execution_space>(0, num_edge),
                         [=](size_t i) {
                           for (int j = 0; j < 3; j++)
                             ptr[i * 3 + j] = mMorseForce(i, j);
                         });
    Kokkos::fence();

    return py_morse_force;
  }

  pybind11::array_t<std::size_t> get_morse_force_by_index(
      pybind11::array_t<std::size_t> _target_index) {
    auto data = _target_index.unchecked<1>();

    std::size_t num_target = (std::size_t)_target_index.shape(0);

    pybind11::array_t<float> py_morse_force(
        {num_target, mMorseForce.extent(1)});

    pybind11::buffer_info buf = py_morse_force.request();
    auto ptr = (float *)buf.ptr;

    Kokkos::parallel_for(
        Kokkos::RangePolicy<host_execution_space>(0, num_target),
        [=](size_t i) {
          for (int j = 0; j < 3; j++)
            ptr[i * 3 + j] = mMorseForce(data(i), j);
        });
    Kokkos::fence();

    return py_morse_force;
  }

  pybind11::array_t<float> get_edge_attr3() {
    pybind11::array_t<float> py_edge_attr3(
        {mThreeBodyEdgeInfo.extent(0), 2 * mMorseForce.extent(1)});

    pybind11::buffer_info buf = py_edge_attr3.request();
    auto ptr = (float *)buf.ptr;

    const int num_edge = mThreeBodyEdgeInfo.extent(0);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<host_execution_space>(0, num_edge), [=](size_t i) {
          for (int j = 0; j < 3; j++)
            ptr[i * 6 + j] = mMorseForce(mThreeBodyEdgeInfo(i, 0), j);

          for (int j = 0; j < 3; j++)
            ptr[i * 6 + j + 3] = mMorseForce(mThreeBodyEdgeInfo(i, 2), j);
        });
    Kokkos::fence();

    return py_edge_attr3;
  }

  pybind11::array_t<float> get_edge_attr_self() {
    pybind11::array_t<float> py_edge_attr_self(
        {mThreeBodyEdgeSelfInfo.extent(0), mMorseForce.extent(1)});

    pybind11::buffer_info buf = py_edge_attr_self.request();
    auto ptr = (float *)buf.ptr;

    const int num_edge = mThreeBodyEdgeSelfInfo.extent(0);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<host_execution_space>(0, num_edge), [=](size_t i) {
          for (int j = 0; j < 3; j++)
            ptr[i * 3 + j] = mMorseForce(mThreeBodyEdgeSelfInfo(i, 1), j);
        });
    Kokkos::fence();

    return py_edge_attr_self;
  }

  pybind11::array_t<float> get_edge_attr() {
    const std::size_t num_edge =
        mTargetSites.extent(0) * (mTargetSites.extent(0) - 1);
    pybind11::array_t<float> py_edge_attr({num_edge, mMorseForce.extent(1)});

    pybind11::buffer_info buf = py_edge_attr.request();
    auto ptr = (float *)buf.ptr;

    Kokkos::parallel_for(
        Kokkos::RangePolicy<host_execution_space>(0, mTargetSites.extent(0)),
        [=](size_t i) {
          for (int j = 0; j < mTargetSites.extent(0) - 1; j++)
            for (int k = 0; k < 3; k++)
              ptr[(i * (mTargetSites.extent(0) - 1) + j) * 3 + k] =
                  mMorseForce(i, k);
        });
    Kokkos::fence();

    return py_edge_attr;
  }

  pybind11::array_t<std::size_t> get_three_body_edge_info() {
    pybind11::array_t<std::size_t> py_edge_info(
        {mThreeBodyEdgeInfo.extent(1), mThreeBodyEdgeInfo.extent(0)});

    pybind11::buffer_info buf = py_edge_info.request();
    auto ptr = (size_t *)buf.ptr;

    const int num_edge = mThreeBodyEdgeInfo.extent(0);
    Kokkos::parallel_for(Kokkos::RangePolicy<host_execution_space>(0, num_edge),
                         [=](size_t i) {
                           ptr[i] = mThreeBodyEdgeInfo(i, 0);
                           ptr[num_edge + i] = mThreeBodyEdgeInfo(i, 1);
                           ptr[2 * num_edge + i] = mThreeBodyEdgeInfo(i, 2);
                         });
    Kokkos::fence();

    return py_edge_info;
  }

  pybind11::array_t<std::size_t> get_three_body_edge_info_by_index(
      pybind11::array_t<std::size_t> _target_index) {
    auto data = _target_index.unchecked<1>();

    std::size_t num_target = (std::size_t)_target_index.shape(0);

    // count edge and calculate offset
    Kokkos::View<size_t *, host_execution_space> edge_local_offset(
        "edge local offset", num_target);
    Kokkos::parallel_scan(
        Kokkos::RangePolicy<host_execution_space>(0, num_target),
        KOKKOS_LAMBDA(const int i, size_t &lsum, bool final) {
          edge_local_offset(i) = lsum;
          lsum += mThreeBodyEdgeNum(data(i));
        });
    Kokkos::fence();

    std::size_t numEdge = edge_local_offset(num_target - 1) +
                          mThreeBodyEdgeNum(data(num_target - 1));

    pybind11::array_t<std::size_t> py_edge_info(
        {mThreeBodyEdgeInfo.extent(1), numEdge});

    pybind11::buffer_info buf = py_edge_info.request();
    auto ptr = (size_t *)buf.ptr;

    Kokkos::parallel_for(
        Kokkos::RangePolicy<host_execution_space>(0, num_target),
        [=](size_t i) {
          unsigned int index = data(i);
          unsigned int offset = mThreeBodyEdgeOffset(index);
          unsigned int num_index_edge = mThreeBodyEdgeNum(index);
          for (unsigned int j = 0; j < num_index_edge; j++) {
            std::size_t counter = edge_local_offset(i) + j;
            ptr[counter] = mThreeBodyEdgeInfo(offset + j, 2);
            ptr[numEdge + counter] = mThreeBodyEdgeInfo(offset + j, 1);
            ptr[2 * numEdge + counter] = mThreeBodyEdgeInfo(offset + j, 0);
          }
        });

    return py_edge_info;
  }

  pybind11::array_t<std::size_t> get_three_body_edge_self_info() {
    pybind11::array_t<std::size_t> py_edge_info(
        {mThreeBodyEdgeSelfInfo.extent(1), mThreeBodyEdgeSelfInfo.extent(0)});

    pybind11::buffer_info buf = py_edge_info.request();
    auto ptr = (size_t *)buf.ptr;

    const int num_edge = mThreeBodyEdgeSelfInfo.extent(0);
    Kokkos::parallel_for(Kokkos::RangePolicy<host_execution_space>(0, num_edge),
                         [=](size_t i) {
                           ptr[i] = mThreeBodyEdgeSelfInfo(i, 0);
                           ptr[num_edge + i] = mThreeBodyEdgeSelfInfo(i, 1);
                         });
    Kokkos::fence();

    return py_edge_info;
  }

  pybind11::array_t<std::size_t> get_three_body_edge_self_info_by_index(
      pybind11::array_t<std::size_t> _target_index) {
    auto data = _target_index.unchecked<1>();

    std::size_t numTarget = (std::size_t)_target_index.shape(0);

    // count edge and calculate offset
    Kokkos::View<size_t *, host_execution_space> edge_self_local_offset(
        "edge self local offset", numTarget);
    Kokkos::parallel_scan(
        Kokkos::RangePolicy<host_execution_space>(0, numTarget),
        KOKKOS_LAMBDA(const int i, size_t &lsum, bool final) {
          edge_self_local_offset(i) = lsum;
          lsum += mThreeBodyEdgeSelfNum(data(i));
        });
    Kokkos::fence();

    std::size_t numEdge = edge_self_local_offset(numTarget - 1) +
                          mThreeBodyEdgeSelfNum(data(numTarget - 1));

    pybind11::array_t<std::size_t> py_edge_self_info(
        {mThreeBodyEdgeSelfInfo.extent(1), numEdge});

    pybind11::buffer_info buf = py_edge_self_info.request();
    auto ptr = (size_t *)buf.ptr;

    Kokkos::parallel_for(
        Kokkos::RangePolicy<host_execution_space>(0, numTarget), [=](size_t i) {
          unsigned int index = data(i);
          unsigned int offset = mThreeBodyEdgeSelfOffset(index);
          unsigned int num_index_edge = mThreeBodyEdgeSelfNum(index);
          for (unsigned int j = 0; j < num_index_edge; j++) {
            std::size_t counter = edge_self_local_offset(i) + j;
            ptr[counter] = mThreeBodyEdgeSelfInfo(offset + j, 1);
            ptr[numEdge + counter] = mThreeBodyEdgeSelfInfo(offset + j, 0);
          }
        });

    return py_edge_self_info;
  }

  pybind11::array_t<float> get_three_body_edge_attr(
      pybind11::array_t<float> _force) {
    pybind11::array_t<float> py_edge_attr({(long)mThreeBodyEdgeInfo.extent(1)});

    pybind11::buffer_info buf = py_edge_attr.request();
    auto ptr = (size_t *)buf.ptr;

    auto data = _force.unchecked<1>();

    Kokkos::parallel_for(Kokkos::RangePolicy<host_execution_space>(
                             0, mThreeBodyEdgeInfo.extent(1)),
                         [&](size_t i) { ptr[i] = data(i); });
    Kokkos::fence();

    return py_edge_attr;
  }

  void build_source_sites() {
    if (!mIsSourceSitesBuilt) {
      if (!mIsPeriodicBoundary) {
        mSourceSites = mTargetSites;

        mSourceIndex =
            decltype(mSourceIndex)("source index", mSourceSites.extent(0));
        Kokkos::parallel_for(Kokkos::RangePolicy<host_execution_space>(
                                 0, mSourceIndex.extent(0)),
                             [&](size_t i) { mSourceIndex(i) = i; });
        Kokkos::fence();
      } else {
        float epsilon = std::max(mTwoBodyEpsilon, mThreeBodyEpsilon);

        float core_domain_low[3], core_domain_high[3], domain_size[3];

        for (int i = 0; i < 3; i++)
          core_domain_low[i] = mDomainLow[i] + epsilon;
        for (int i = 0; i < 3; i++)
          core_domain_high[i] = mDomainHigh[i] - epsilon;
        for (int i = 0; i < 3; i++)
          domain_size[i] = mDomainHigh[i] - mDomainLow[i];

        const std::size_t num_target = mTargetSites.extent(0);

        HostIndexVector num_source_duplicate =
            HostIndexVector("num source duplicate", num_target);
        HostIntMatrix axis_source_duplicate =
            HostIntMatrix("axis source duplicate", num_target, 3);

        Kokkos::parallel_for(
            Kokkos::RangePolicy<host_execution_space>(0, num_target),
            [&](size_t i) {
              std::size_t num = 1;
              for (int j = 0; j < 3; j++) {
                axis_source_duplicate(i, j) = 1;
                if (mTargetSites(i, j) < core_domain_low[j]) {
                  axis_source_duplicate(i, j) = -2;
                  num *= 2;
                } else if (mTargetSites(i, j) > core_domain_high[j]) {
                  axis_source_duplicate(i, j) = 2;
                  num *= 2;
                } else {
                  axis_source_duplicate(i, j) = 1;
                }
              }

              num_source_duplicate(i) = num;
            });
        Kokkos::fence();

        HostIndexVector num_source_offset =
            HostIndexVector("num source offset", num_target + 1);

        num_source_offset(0) = 0;
        for (int i = 0; i < num_target; i++) {
          num_source_offset(i + 1) =
              num_source_offset(i) + num_source_duplicate(i);
        }
        std::size_t num_source = num_source_offset(num_target);

        mSourceSites = decltype(mSourceSites)("source sites", num_source, 3);
        mSourceIndex = decltype(mSourceIndex)("source index", num_source);

        Kokkos::parallel_for(
            Kokkos::RangePolicy<host_execution_space>(0, num_target),
            [&](size_t i) {
              std::vector<float> offset;
              offset.resize(3 * num_source_duplicate(i));

              const std::size_t num = num_source_duplicate(i);
              std::size_t stride1 = num_source_duplicate(i);
              for (int j = 0; j < 3; j++) {
                if (axis_source_duplicate(i, j) == 1) {
                  for (int n = 0; n < num; n++) {
                    offset[n * 3 + j] = 0;
                  }
                }
                if (axis_source_duplicate(i, j) == 2) {
                  for (int m = 0; m < num; m += stride1) {
                    for (int n = m; n < m + stride1 / 2; n++) {
                      offset[n * 3 + j] = 0;
                    }
                    for (int n = m + stride1 / 2; n < m + stride1; n++) {
                      offset[n * 3 + j] = -domain_size[j];
                    }
                  }
                  stride1 /= 2;
                }
                if (axis_source_duplicate(i, j) == -2) {
                  for (int m = 0; m < num; m += stride1) {
                    for (int n = m; n < m + stride1 / 2; n++) {
                      offset[n * 3 + j] = 0;
                    }
                    for (int n = m + stride1 / 2; n < m + stride1; n++) {
                      offset[n * 3 + j] = domain_size[j];
                    }
                  }
                  stride1 /= 2;
                }
              }

              for (int m = num_source_offset[i]; m < num_source_offset[i + 1];
                   m++) {
                for (int j = 0; j < 3; j++) {
                  mSourceSites(m, j) =
                      mTargetSites(i, j) +
                      offset[(m - num_source_offset[i]) * 3 + j];
                }
                mSourceIndex(m) = i;
              }
            });
        Kokkos::fence();
      }

      mPointCloudSearch = std::make_shared<PointCloudSearch<HostFloatMatrix>>(
          CreatePointCloudSearch(mSourceSites, 3));
    }

    mIsSourceSitesBuilt = true;
  }

  void build_target_neighbor_lists(float epsilon) {
    build_source_sites();

    auto num_target = mTargetSites.extent(0);

    Kokkos::parallel_for(
        Kokkos::RangePolicy<host_execution_space>(0, num_target),
        [=](size_t i) { mEpsilonLists(i) = epsilon; });
    Kokkos::fence();

    auto num_neighbor =
        1 + mPointCloudSearch->generate2DNeighborListsFromRadiusSearch(
                true, mTargetSites, mTargetNeighborLists, mEpsilonLists, 0.0,
                epsilon);
    if (num_neighbor > mTargetNeighborLists.extent(1))
      Kokkos::resize(mTargetNeighborLists, num_target, num_neighbor);

    mPointCloudSearch->generate2DNeighborListsFromRadiusSearch(
        false, mTargetSites, mTargetNeighborLists, mEpsilonLists, 0.0, epsilon);

    Kokkos::parallel_for(
        Kokkos::RangePolicy<host_execution_space>(0, num_target),
        [=](size_t i) {
          // change the target index to the real index
          int counter = 0;
          while (counter < mTargetNeighborLists(i, 0)) {
            mTargetNeighborLists(i, counter + 1) =
                mSourceIndex(mTargetNeighborLists(i, counter + 1));
            counter++;
          }
        });
    Kokkos::fence();

    Kokkos::parallel_for(
        Kokkos::RangePolicy<host_execution_space>(0, num_target),
        [=](size_t i) {
          // ensure the current target index appears at the
          // beginning of the list
          int counter = 0;
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

  void build_two_body_info() {
    build_target_neighbor_lists(mTwoBodyEpsilon);

    std::size_t numTarget = mTargetSites.extent(0);

    Kokkos::resize(mTwoBodyEdgeNum, numTarget);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<host_execution_space>(0, numTarget),
        [=](std::size_t i) {
          mTwoBodyEdgeNum(i) = mTargetNeighborLists(i, 0) - 1;
        });
    Kokkos::fence();

    Kokkos::resize(mTwoBodyEdgeOffset, numTarget);

    Kokkos::parallel_scan(
        Kokkos::RangePolicy<host_execution_space>(0, numTarget),
        KOKKOS_LAMBDA(const int i, size_t &lsum, bool final) {
          mTwoBodyEdgeOffset(i) = lsum;
          lsum += mTwoBodyEdgeNum(i);
        });
    Kokkos::fence();

    const int num_edge =
        mTwoBodyEdgeOffset(numTarget - 1) + mTwoBodyEdgeNum(numTarget - 1);

    mTwoBodyEdgeInfo =
        decltype(mTwoBodyEdgeInfo)("two body edge info", num_edge, 2);
    mMorseForce = decltype(mMorseForce)("morse force", numTarget, 3);

    Kokkos::parallel_for(
        Kokkos::RangePolicy<host_execution_space>(0, numTarget), [=](size_t i) {
          std::size_t offset = 0;
          for (int j = 0; j < 3; j++)
            mMorseForce(i, j) = 0.0;

          for (std::size_t j = 1; j < mTargetNeighborLists(i, 0); j++) {
            std::size_t neighbor_idx = mTargetNeighborLists(i, j + 1);
            mTwoBodyEdgeInfo(mTwoBodyEdgeOffset(i) + offset, 0) = i;
            mTwoBodyEdgeInfo(mTwoBodyEdgeOffset(i) + offset, 1) =
                mSourceIndex(neighbor_idx);

            float r[3], f[3];
            for (int k = 0; k < 3; k++)
              r[k] = mTargetSites(i, k) - mSourceSites(neighbor_idx, k);

            calculate_morse(r, f);
            for (int k = 0; k < 3; k++)
              mMorseForce(i, k) += f[k];

            offset++;
          }
        });
    Kokkos::fence();
  }

  void build_three_body_info() {
    build_target_neighbor_lists(mThreeBodyEpsilon);

    std::size_t numTarget = mTargetSites.extent(0);

    Kokkos::resize(mThreeBodyEdgeNum, numTarget);
    Kokkos::resize(mThreeBodyEdgeSelfNum, numTarget);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<host_execution_space>(0, numTarget),
        [=](std::size_t i) {
          mThreeBodyEdgeNum(i) = 0;
          mThreeBodyEdgeSelfNum(i) = mTargetNeighborLists(i, 0) - 1;
          for (std::size_t j = 1; j < mTargetNeighborLists(i, 0); j++) {
            int neighbor_idx = mTargetNeighborLists(i, j + 1);
            mThreeBodyEdgeNum(i) += (mTargetNeighborLists(neighbor_idx, 0) - 2);
          }
        });
    Kokkos::fence();

    Kokkos::resize(mThreeBodyEdgeOffset, numTarget);
    Kokkos::resize(mThreeBodyEdgeSelfOffset, numTarget);

    Kokkos::parallel_scan(
        Kokkos::RangePolicy<host_execution_space>(0, numTarget),
        KOKKOS_LAMBDA(const int i, size_t &lsum, bool final) {
          mThreeBodyEdgeOffset(i) = lsum;
          lsum += mThreeBodyEdgeNum(i);
        });
    Kokkos::fence();
    Kokkos::parallel_scan(
        Kokkos::RangePolicy<host_execution_space>(0, numTarget),
        KOKKOS_LAMBDA(const int i, size_t &lsum, bool final) {
          mThreeBodyEdgeSelfOffset(i) = lsum;
          lsum += mThreeBodyEdgeSelfNum(i);
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
        Kokkos::RangePolicy<host_execution_space>(0, numTarget), [=](size_t i) {
          std::size_t offset = 0;
          for (std::size_t j = 1; j < mTargetNeighborLists(i, 0); j++) {
            std::size_t neighbor_idx = mTargetNeighborLists(i, j + 1);
            for (std::size_t it = 1; it < mTargetNeighborLists(neighbor_idx, 0);
                 it++) {
              std::size_t neighbor_neighbor_idx =
                  mTargetNeighborLists(neighbor_idx, it + 1);
              if (neighbor_neighbor_idx != i) {
                mThreeBodyEdgeInfo(mThreeBodyEdgeOffset(i) + offset, 0) = i;
                mThreeBodyEdgeInfo(mThreeBodyEdgeOffset(i) + offset, 1) =
                    neighbor_idx;
                mThreeBodyEdgeInfo(mThreeBodyEdgeOffset(i) + offset, 2) =
                    neighbor_neighbor_idx;

                offset++;
              }
            }

            mThreeBodyEdgeSelfInfo(mThreeBodyEdgeSelfOffset(i) + j - 1, 0) = i;
            mThreeBodyEdgeSelfInfo(mThreeBodyEdgeSelfOffset(i) + j - 1, 1) =
                neighbor_idx;
          }
        });
    Kokkos::fence();
  }
};

#endif