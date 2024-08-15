#ifndef target_neighbor_lists_HPP
#define target_neighbor_lists_HPP

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

#include <Compadre_PointCloudSearch.hpp>

// TODO: morse_force and three_body_edge_attr if it is selected by index, it is
// still problematic

class neighbor_lists {
public:
  typedef Kokkos::View<float **, host_execution_space> site_view_type;
  typedef Kokkos::View<float *, host_execution_space> real_view_type;
  typedef Kokkos::View<std::size_t **, host_execution_space> view_type;
  typedef Kokkos::View<std::size_t *, host_execution_space> index_view_type;
  typedef Kokkos::View<int **, host_execution_space> int_view_type;

private:
  site_view_type source_sites;
  site_view_type target_sites;
  site_view_type morse_force;
  real_view_type epsilon_lists;
  index_view_type source_index;
  view_type two_body_edge_info;
  view_type three_body_edge_info;
  view_type three_body_edge_self_info;
  view_type target_neighbor_lists;

  index_view_type two_body_edge_num;
  index_view_type three_body_edge_num;
  index_view_type three_body_edge_self_num;
  index_view_type two_body_edge_offset;
  index_view_type three_body_edge_offset;
  index_view_type three_body_edge_self_offset;

  bool is_source_sites_built;
  bool is_periodic_boundary;
  float two_body_epsilon, three_body_epsilon;
  float domain_low[3], domain_high[3];
  int dim;

  std::shared_ptr<Compadre::PointCloudSearch<site_view_type>>
      point_cloud_search;

public:
  neighbor_lists()
      : is_source_sites_built(false), is_periodic_boundary(false), dim(3),
        two_body_epsilon(0.0), three_body_epsilon(0.0) {}

  ~neighbor_lists() {}

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
    is_periodic_boundary = _is_periodic_boundary;
  }

  void set_domain(pybind11::array_t<float> _domain) {
    pybind11::buffer_info buf = _domain.request();

    if (buf.ndim != 2) {
      throw std::runtime_error("Rank of domain must be two");
    }

    auto data = _domain.unchecked<2>();

    domain_low[0] = data(0, 0);
    domain_low[1] = data(0, 1);
    domain_low[2] = data(0, 2);

    domain_high[0] = data(1, 0);
    domain_high[1] = data(1, 1);
    domain_high[2] = data(1, 2);
  }

  void set_dimension(int _dim) { dim = _dim; }

  void set_target_sites(pybind11::array_t<float> _target_sites) {
    pybind11::buffer_info buf = _target_sites.request();

    if (buf.ndim != 2) {
      throw std::runtime_error("Rank of target sites must be two");
    }

    target_sites = decltype(target_sites)("target sites",
                                          (std::size_t)_target_sites.shape(0),
                                          (std::size_t)_target_sites.shape(1));

    auto data = _target_sites.unchecked<2>();

    Kokkos::parallel_for(
        Kokkos::RangePolicy<host_execution_space>(0, target_sites.extent(0)),
        [=](size_t i) {
          target_sites(i, 0) = data(i, 0);
          target_sites(i, 1) = data(i, 1);
          target_sites(i, 2) = data(i, 2);
        });
    Kokkos::fence();

    target_neighbor_lists = decltype(target_neighbor_lists)(
        "two body neighbor lists", target_sites.extent(0), 2);
    target_neighbor_lists = decltype(target_neighbor_lists)(
        "three body neighbor lists", target_sites.extent(0), 2);
    epsilon_lists =
        decltype(epsilon_lists)("epsilon lists", target_sites.extent(0));

    is_source_sites_built = false;
  }

  void set_two_body_epsilon(float _epsilon) { two_body_epsilon = _epsilon; }

  void set_three_body_epsilon(float _epsilon) { three_body_epsilon = _epsilon; }

  pybind11::array_t<std::size_t> get_two_body_edge_info() {
    pybind11::array_t<std::size_t> py_edge_info(
        {two_body_edge_info.extent(1), two_body_edge_info.extent(0)});

    pybind11::buffer_info buf = py_edge_info.request();
    auto ptr = (size_t *)buf.ptr;

    const int num_edge = two_body_edge_info.extent(0);
    Kokkos::parallel_for(Kokkos::RangePolicy<host_execution_space>(0, num_edge),
                         [=](size_t i) { ptr[i] = two_body_edge_info(i, 0); });
    Kokkos::fence();
    Kokkos::parallel_for(
        Kokkos::RangePolicy<host_execution_space>(0, num_edge),
        [=](size_t i) { ptr[num_edge + i] = two_body_edge_info(i, 1); });
    Kokkos::fence();

    return py_edge_info;
  }

  pybind11::array_t<std::size_t> get_two_body_edge_info_by_index(
      pybind11::array_t<std::size_t> _target_index) {
    auto data = _target_index.unchecked<1>();

    std::size_t num_target = (std::size_t)_target_index.shape(0);

    std::size_t totalTarget = target_sites.extent(0);
    std::size_t edgePerTarget = target_sites.extent(0) - 1;
    std::size_t numEdge = num_target * edgePerTarget;

    pybind11::array_t<std::size_t> py_edge_info(
        {two_body_edge_info.extent(1), numEdge});

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
    //       lsum += two_body_edge_num(data(i));
    //     });
    // Kokkos::fence();

    // std::size_t numEdge = edge_local_offset(num_target - 1) +
    //                       two_body_edge_num(data(num_target - 1));

    // pybind11::array_t<std::size_t> py_edge_info(
    //     {two_body_edge_info.extent(1), numEdge});

    // pybind11::buffer_info buf = py_edge_info.request();
    // auto ptr = (size_t *)buf.ptr;

    // Kokkos::parallel_for(
    //     Kokkos::RangePolicy<host_execution_space>(0, num_target),
    //     [=](size_t i) {
    //       unsigned int index = data(i);
    //       unsigned int offset = two_body_edge_offset(index);
    //       unsigned int num_index_edge = two_body_edge_num(index);
    //       for (unsigned int j = 0; j < num_index_edge; j++) {
    //         std::size_t counter = edge_local_offset(i) + j;
    //         ptr[counter] = two_body_edge_info(offset + j, 0);
    //         ptr[numEdge + counter] = two_body_edge_info(offset + j, 1);
    //       }
    //     });

    return py_edge_info;
  }

  pybind11::array_t<float> get_morse_force() {
    pybind11::array_t<float> py_morse_force(
        {morse_force.extent(0), morse_force.extent(1)});

    pybind11::buffer_info buf = py_morse_force.request();
    auto ptr = (float *)buf.ptr;

    const int num_edge = morse_force.extent(0);
    Kokkos::parallel_for(Kokkos::RangePolicy<host_execution_space>(0, num_edge),
                         [=](size_t i) {
                           for (int j = 0; j < 3; j++)
                             ptr[i * 3 + j] = morse_force(i, j);
                         });
    Kokkos::fence();

    return py_morse_force;
  }

  pybind11::array_t<std::size_t>
  get_morse_force_by_index(pybind11::array_t<std::size_t> _target_index) {
    auto data = _target_index.unchecked<1>();

    std::size_t num_target = (std::size_t)_target_index.shape(0);

    pybind11::array_t<float> py_morse_force(
        {num_target, morse_force.extent(1)});

    pybind11::buffer_info buf = py_morse_force.request();
    auto ptr = (float *)buf.ptr;

    Kokkos::parallel_for(
        Kokkos::RangePolicy<host_execution_space>(0, num_target),
        [=](size_t i) {
          for (int j = 0; j < 3; j++)
            ptr[i * 3 + j] = morse_force(data(i), j);
        });
    Kokkos::fence();

    return py_morse_force;
  }

  pybind11::array_t<float> get_edge_attr3() {
    pybind11::array_t<float> py_edge_attr3(
        {three_body_edge_info.extent(0), 2 * morse_force.extent(1)});

    pybind11::buffer_info buf = py_edge_attr3.request();
    auto ptr = (float *)buf.ptr;

    const int num_edge = three_body_edge_info.extent(0);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<host_execution_space>(0, num_edge), [=](size_t i) {
          for (int j = 0; j < 3; j++)
            ptr[i * 6 + j] = morse_force(three_body_edge_info(i, 0), j);

          for (int j = 0; j < 3; j++)
            ptr[i * 6 + j + 3] = morse_force(three_body_edge_info(i, 2), j);
        });
    Kokkos::fence();

    return py_edge_attr3;
  }

  pybind11::array_t<float> get_edge_attr_self() {
    pybind11::array_t<float> py_edge_attr_self(
        {three_body_edge_self_info.extent(0), morse_force.extent(1)});

    pybind11::buffer_info buf = py_edge_attr_self.request();
    auto ptr = (float *)buf.ptr;

    const int num_edge = three_body_edge_self_info.extent(0);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<host_execution_space>(0, num_edge), [=](size_t i) {
          for (int j = 0; j < 3; j++)
            ptr[i * 3 + j] = morse_force(three_body_edge_self_info(i, 1), j);
        });
    Kokkos::fence();

    return py_edge_attr_self;
  }

  pybind11::array_t<float> get_edge_attr() {
    const std::size_t num_edge =
        target_sites.extent(0) * (target_sites.extent(0) - 1);
    pybind11::array_t<float> py_edge_attr({num_edge, morse_force.extent(1)});

    pybind11::buffer_info buf = py_edge_attr.request();
    auto ptr = (float *)buf.ptr;

    Kokkos::parallel_for(
        Kokkos::RangePolicy<host_execution_space>(0, target_sites.extent(0)),
        [=](size_t i) {
          for (int j = 0; j < target_sites.extent(0) - 1; j++)
            for (int k = 0; k < 3; k++)
              ptr[(i * (target_sites.extent(0) - 1) + j) * 3 + k] =
                  morse_force(i, k);
        });
    Kokkos::fence();

    return py_edge_attr;
  }

  pybind11::array_t<std::size_t> get_three_body_edge_info() {
    pybind11::array_t<std::size_t> py_edge_info(
        {three_body_edge_info.extent(1), three_body_edge_info.extent(0)});

    pybind11::buffer_info buf = py_edge_info.request();
    auto ptr = (size_t *)buf.ptr;

    const int num_edge = three_body_edge_info.extent(0);
    Kokkos::parallel_for(Kokkos::RangePolicy<host_execution_space>(0, num_edge),
                         [=](size_t i) {
                           ptr[i] = three_body_edge_info(i, 0);
                           ptr[num_edge + i] = three_body_edge_info(i, 1);
                           ptr[2 * num_edge + i] = three_body_edge_info(i, 2);
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
          lsum += three_body_edge_num(data(i));
        });
    Kokkos::fence();

    std::size_t numEdge = edge_local_offset(num_target - 1) +
                          three_body_edge_num(data(num_target - 1));

    pybind11::array_t<std::size_t> py_edge_info(
        {three_body_edge_info.extent(1), numEdge});

    pybind11::buffer_info buf = py_edge_info.request();
    auto ptr = (size_t *)buf.ptr;

    Kokkos::parallel_for(
        Kokkos::RangePolicy<host_execution_space>(0, num_target),
        [=](size_t i) {
          unsigned int index = data(i);
          unsigned int offset = three_body_edge_offset(index);
          unsigned int num_index_edge = three_body_edge_num(index);
          for (unsigned int j = 0; j < num_index_edge; j++) {
            std::size_t counter = edge_local_offset(i) + j;
            ptr[counter] = three_body_edge_info(offset + j, 2);
            ptr[numEdge + counter] = three_body_edge_info(offset + j, 1);
            ptr[2 * numEdge + counter] = three_body_edge_info(offset + j, 0);
          }
        });

    return py_edge_info;
  }

  pybind11::array_t<std::size_t> get_three_body_edge_self_info() {
    pybind11::array_t<std::size_t> py_edge_info(
        {three_body_edge_self_info.extent(1),
         three_body_edge_self_info.extent(0)});

    pybind11::buffer_info buf = py_edge_info.request();
    auto ptr = (size_t *)buf.ptr;

    const int num_edge = three_body_edge_self_info.extent(0);
    Kokkos::parallel_for(Kokkos::RangePolicy<host_execution_space>(0, num_edge),
                         [=](size_t i) {
                           ptr[i] = three_body_edge_self_info(i, 0);
                           ptr[num_edge + i] = three_body_edge_self_info(i, 1);
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
          lsum += three_body_edge_self_num(data(i));
        });
    Kokkos::fence();

    std::size_t numEdge = edge_self_local_offset(numTarget - 1) +
                          three_body_edge_self_num(data(numTarget - 1));

    pybind11::array_t<std::size_t> py_edge_self_info(
        {three_body_edge_self_info.extent(1), numEdge});

    pybind11::buffer_info buf = py_edge_self_info.request();
    auto ptr = (size_t *)buf.ptr;

    Kokkos::parallel_for(
        Kokkos::RangePolicy<host_execution_space>(0, numTarget), [=](size_t i) {
          unsigned int index = data(i);
          unsigned int offset = three_body_edge_self_offset(index);
          unsigned int num_index_edge = three_body_edge_self_num(index);
          for (unsigned int j = 0; j < num_index_edge; j++) {
            std::size_t counter = edge_self_local_offset(i) + j;
            ptr[counter] = three_body_edge_self_info(offset + j, 1);
            ptr[numEdge + counter] = three_body_edge_self_info(offset + j, 0);
          }
        });

    return py_edge_self_info;
  }

  pybind11::array_t<float>
  get_three_body_edge_attr(pybind11::array_t<float> _force) {
    pybind11::array_t<float> py_edge_attr({three_body_edge_info.extent(1)});

    pybind11::buffer_info buf = py_edge_attr.request();
    auto ptr = (size_t *)buf.ptr;

    auto data = _force.unchecked<1>();

    Kokkos::parallel_for(Kokkos::RangePolicy<host_execution_space>(
                             0, three_body_edge_info.extent(1)),
                         [&](size_t i) { ptr[i] = data(i); });
    Kokkos::fence();

    return py_edge_attr;
  }

  void build_source_sites() {
    if (!is_source_sites_built) {
      if (!is_periodic_boundary) {
        source_sites = target_sites;

        source_index =
            decltype(source_index)("source index", source_sites.extent(0));
        Kokkos::parallel_for(Kokkos::RangePolicy<host_execution_space>(
                                 0, source_index.extent(0)),
                             [&](size_t i) { source_index(i) = i; });
        Kokkos::fence();
      } else {
        float epsilon = std::max(two_body_epsilon, three_body_epsilon);

        float core_domain_low[3], core_domain_high[3], domain_size[3];

        for (int i = 0; i < 3; i++)
          core_domain_low[i] = domain_low[i] + epsilon;
        for (int i = 0; i < 3; i++)
          core_domain_high[i] = domain_high[i] - epsilon;
        for (int i = 0; i < 3; i++)
          domain_size[i] = domain_high[i] - domain_low[i];

        const std::size_t num_target = target_sites.extent(0);

        index_view_type num_source_duplicate =
            index_view_type("num source duplicate", num_target);
        int_view_type axis_source_duplicate =
            int_view_type("axis source duplicate", num_target, 3);

        Kokkos::parallel_for(
            Kokkos::RangePolicy<host_execution_space>(0, num_target),
            [&](size_t i) {
              std::size_t num = 1;
              for (int j = 0; j < 3; j++) {
                axis_source_duplicate(i, j) = 1;
                if (target_sites(i, j) < core_domain_low[j]) {
                  axis_source_duplicate(i, j) = -2;
                  num *= 2;
                } else if (target_sites(i, j) > core_domain_high[j]) {
                  axis_source_duplicate(i, j) = 2;
                  num *= 2;
                } else {
                  axis_source_duplicate(i, j) = 1;
                }
              }

              num_source_duplicate(i) = num;
            });
        Kokkos::fence();

        index_view_type num_source_offset =
            index_view_type("num source offset", num_target + 1);

        num_source_offset(0) = 0;
        for (int i = 0; i < num_target; i++) {
          num_source_offset(i + 1) =
              num_source_offset(i) + num_source_duplicate(i);
        }
        std::size_t num_source = num_source_offset(num_target);

        source_sites = decltype(source_sites)("source sites", num_source, 3);
        source_index = decltype(source_index)("source index", num_source);

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
                  source_sites(m, j) =
                      target_sites(i, j) +
                      offset[(m - num_source_offset[i]) * 3 + j];
                }
                source_index(m) = i;
              }
            });
        Kokkos::fence();
      }

      point_cloud_search =
          std::make_shared<Compadre::PointCloudSearch<site_view_type>>(
              Compadre::CreatePointCloudSearch(source_sites, 3));
    }

    is_source_sites_built = true;
  }

  void build_target_neighbor_lists(float epsilon) {
    build_source_sites();

    auto num_target = target_sites.extent(0);

    Kokkos::parallel_for(
        Kokkos::RangePolicy<host_execution_space>(0, num_target),
        [=](size_t i) { epsilon_lists(i) = epsilon; });
    Kokkos::fence();

    auto num_neighbor =
        1 + point_cloud_search->generate2DNeighborListsFromRadiusSearch(
                true, target_sites, target_neighbor_lists, epsilon_lists, 0.0,
                epsilon);
    if (num_neighbor > target_neighbor_lists.extent(1))
      Kokkos::resize(target_neighbor_lists, num_target, num_neighbor);

    point_cloud_search->generate2DNeighborListsFromRadiusSearch(
        false, target_sites, target_neighbor_lists, epsilon_lists, 0.0,
        epsilon);

    Kokkos::parallel_for(
        Kokkos::RangePolicy<host_execution_space>(0, num_target),
        [=](size_t i) {
          // change the target index to the real index
          int counter = 0;
          while (counter < target_neighbor_lists(i, 0)) {
            target_neighbor_lists(i, counter + 1) =
                source_index(target_neighbor_lists(i, counter + 1));
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
          while (counter < target_neighbor_lists(i, 0)) {
            if (target_neighbor_lists(i, counter + 1) == i) {
              std::swap(target_neighbor_lists(i, 1),
                        target_neighbor_lists(i, counter + 1));
              break;
            }
            counter++;
          }
        });
    Kokkos::fence();
  }

  void build_two_body_info() {
    build_target_neighbor_lists(two_body_epsilon);

    std::size_t numTarget = target_sites.extent(0);

    Kokkos::resize(two_body_edge_num, numTarget);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<host_execution_space>(0, numTarget),
        [=](std::size_t i) {
          two_body_edge_num(i) = target_neighbor_lists(i, 0) - 1;
        });
    Kokkos::fence();

    Kokkos::resize(two_body_edge_offset, numTarget);

    Kokkos::parallel_scan(
        Kokkos::RangePolicy<host_execution_space>(0, numTarget),
        KOKKOS_LAMBDA(const int i, size_t &lsum, bool final) {
          two_body_edge_offset(i) = lsum;
          lsum += two_body_edge_num(i);
        });
    Kokkos::fence();

    const int num_edge =
        two_body_edge_offset(numTarget - 1) + two_body_edge_num(numTarget - 1);

    two_body_edge_info =
        decltype(two_body_edge_info)("two body edge info", num_edge, 2);
    morse_force = decltype(morse_force)("morse force", numTarget, 3);

    Kokkos::parallel_for(
        Kokkos::RangePolicy<host_execution_space>(0, numTarget), [=](size_t i) {
          std::size_t offset = 0;
          for (int j = 0; j < 3; j++)
            morse_force(i, j) = 0.0;

          for (std::size_t j = 1; j < target_neighbor_lists(i, 0); j++) {
            std::size_t neighbor_idx = target_neighbor_lists(i, j + 1);
            two_body_edge_info(two_body_edge_offset(i) + offset, 0) = i;
            two_body_edge_info(two_body_edge_offset(i) + offset, 1) =
                source_index(neighbor_idx);

            float r[3], f[3];
            for (int k = 0; k < 3; k++)
              r[k] = target_sites(i, k) - source_sites(neighbor_idx, k);

            calculate_morse(r, f);
            for (int k = 0; k < 3; k++)
              morse_force(i, k) += f[k];

            offset++;
          }
        });
    Kokkos::fence();
  }

  void build_three_body_info() {
    build_target_neighbor_lists(three_body_epsilon);

    std::size_t numTarget = target_sites.extent(0);

    Kokkos::resize(three_body_edge_num, numTarget);
    Kokkos::resize(three_body_edge_self_num, numTarget);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<host_execution_space>(0, numTarget),
        [=](std::size_t i) {
          three_body_edge_num(i) = 0;
          three_body_edge_self_num(i) = target_neighbor_lists(i, 0) - 1;
          for (std::size_t j = 1; j < target_neighbor_lists(i, 0); j++) {
            int neighbor_idx = target_neighbor_lists(i, j + 1);
            three_body_edge_num(i) +=
                (target_neighbor_lists(neighbor_idx, 0) - 2);
          }
        });
    Kokkos::fence();

    Kokkos::resize(three_body_edge_offset, numTarget);
    Kokkos::resize(three_body_edge_self_offset, numTarget);

    Kokkos::parallel_scan(
        Kokkos::RangePolicy<host_execution_space>(0, numTarget),
        KOKKOS_LAMBDA(const int i, size_t &lsum, bool final) {
          three_body_edge_offset(i) = lsum;
          lsum += three_body_edge_num(i);
        });
    Kokkos::fence();
    Kokkos::parallel_scan(
        Kokkos::RangePolicy<host_execution_space>(0, numTarget),
        KOKKOS_LAMBDA(const int i, size_t &lsum, bool final) {
          three_body_edge_self_offset(i) = lsum;
          lsum += three_body_edge_self_num(i);
        });
    Kokkos::fence();

    Kokkos::resize(three_body_edge_info,
                   three_body_edge_offset(numTarget - 1) +
                       three_body_edge_num(numTarget - 1),
                   3);
    Kokkos::resize(three_body_edge_self_info,
                   three_body_edge_self_offset(numTarget - 1) +
                       three_body_edge_self_num(numTarget - 1),
                   2);

    Kokkos::parallel_for(
        Kokkos::RangePolicy<host_execution_space>(0, numTarget), [=](size_t i) {
          std::size_t offset = 0;
          for (std::size_t j = 1; j < target_neighbor_lists(i, 0); j++) {
            std::size_t neighbor_idx = target_neighbor_lists(i, j + 1);
            for (std::size_t it = 1;
                 it < target_neighbor_lists(neighbor_idx, 0); it++) {
              std::size_t neighbor_neighbor_idx =
                  target_neighbor_lists(neighbor_idx, it + 1);
              if (neighbor_neighbor_idx != i) {
                three_body_edge_info(three_body_edge_offset(i) + offset, 0) = i;
                three_body_edge_info(three_body_edge_offset(i) + offset, 1) =
                    neighbor_idx;
                three_body_edge_info(three_body_edge_offset(i) + offset, 2) =
                    neighbor_neighbor_idx;

                offset++;
              }
            }

            three_body_edge_self_info(three_body_edge_self_offset(i) + j - 1,
                                      0) = i;
            three_body_edge_self_info(three_body_edge_self_offset(i) + j - 1,
                                      1) = neighbor_idx;
          }
        });
    Kokkos::fence();
  }
};

#endif
