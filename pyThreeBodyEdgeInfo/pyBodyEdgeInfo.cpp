#include "kokkos_interface.hpp"
#include "neighbor_lists.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

PYBIND11_MODULE(pyBodyEdgeInfo, m) {
  m.doc() = R"pbdoc(
        Body edge detection for Python
    )pbdoc";

  py::class_<kokkos_interface>(m, "KokkosInterface").def(py::init());

  py::class_<neighbor_lists>(m, "BodyEdgeInfo")
      .def(py::init())
      .def("setTargetSites", overload_cast_<pybind11::array_t<float>>()(
                                 &neighbor_lists::set_target_sites))
      .def("setPeriodic", overload_cast_<bool>()(&neighbor_lists::set_periodic))
      .def("setDomain", overload_cast_<pybind11::array_t<float>>()(
                            &neighbor_lists::set_domain))
      .def("setTwoBodyEpsilon", &neighbor_lists::set_two_body_epsilon)
      .def("setThreeBodyEpsilon", &neighbor_lists::set_three_body_epsilon)
      .def("buildTwoBodyEdgeInfo", &neighbor_lists::build_two_body_info)
      .def("buildThreeBodyEdgeInfo", &neighbor_lists::build_three_body_info)
      .def("getThreeBodyEdgeInfo", &neighbor_lists::get_three_body_edge_info)
      .def("getThreeBodyEdgeInfoByIndex",
           overload_cast_<pybind11::array_t<std::size_t>>()(
               &neighbor_lists::get_three_body_edge_info_by_index))
      .def("getThreeBodyEdgeSelfInfo",
           &neighbor_lists::get_three_body_edge_self_info)
      .def("getThreeBodyEdgeSelfInfoByIndex",
           overload_cast_<pybind11::array_t<std::size_t>>()(
               &neighbor_lists::get_three_body_edge_self_info_by_index))
      .def("getTwoBodyEdgeInfo", &neighbor_lists::get_two_body_edge_info)
      .def("getTwoBodyEdgeInfoByIndex",
           overload_cast_<pybind11::array_t<std::size_t>>()(
               &neighbor_lists::get_two_body_edge_info_by_index))
      .def("getMorseForce", &neighbor_lists::get_morse_force)
      .def("getMorseForceByIndex",
           overload_cast_<pybind11::array_t<std::size_t>>()(
               &neighbor_lists::get_morse_force_by_index))
      .def("getEdgeAttr3", &neighbor_lists::get_edge_attr3)
      .def("getEdgeAttrSelf", &neighbor_lists::get_edge_attr_self)
      .def("getEdgeAttr", &neighbor_lists::get_edge_attr)
      .def("getThreeBodyEdgeAttr",
           overload_cast_<pybind11::array_t<float>>()(
               &neighbor_lists::get_three_body_edge_attr));
}