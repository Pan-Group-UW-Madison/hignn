#include "HignnModel.hpp"
#include "TimeIntegrator.hpp"
#include "ThreeBodyEdge.hpp"

template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

PYBIND11_MODULE(hignn, m) {
  m.doc() = R"pbdoc(
        Hydrodynamic Interaction Graph Neural Network
    )pbdoc";

  m.def("Init", &Init, "Initialize MPI and Kokkos");
  m.def("Finalize", &Finalize, "Finalize MPI and Kokkos");

  py::class_<HignnModel>(m, "HignnModel")
      .def(py::init<pybind11::array_t<float> &, int>())
      .def("LoadTwoBodyModel", &HignnModel::LoadTwoBodyModel)
      .def("LoadThreeBodyModel", &HignnModel::LoadThreeBodyModel)
      .def("SetEpsilon", &HignnModel::SetEpsilon)
      .def("SetMaxIter", &HignnModel::SetMaxIter)
      .def("SetMatPoolSizeFactor", &HignnModel::SetMatPoolSizeFactor)
      .def("SetPostCheckFlag", &HignnModel::SetPostCheckFlag)
      .def("Update", &HignnModel::Update)
      .def("UpdateCoord", &HignnModel::UpdateCoord)
      .def("Dot", &HignnModel::Dot);

  py::class_<ExplicitEuler>(m, "ExplicitEuler")
      .def(py::init())
      .def("setTimeStep", &ExplicitEuler::set_time_step)
      .def("setFinalTime", &ExplicitEuler::set_final_time)
      .def("setNumRigidBody", &ExplicitEuler::set_num_rigid_body)
      .def("setOutputStep", &ExplicitEuler::set_output_step)
      .def("initialize", &ExplicitEuler::init)
      .def("setVelocityFunc", &ExplicitEuler::set_python_velocity_update_func)
      .def("setXLim", &ExplicitEuler::set_xlim)
      .def("setYLim", &ExplicitEuler::set_ylim)
      .def("setZLim", &ExplicitEuler::set_zlim)
      .def("run", &ExplicitEuler::run);

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

  //   py::class_<explicit_RK4>(m, "ExplicitRK4")
  //       .def(py::init())
  //       .def("setTimeStep", &explicit_RK4::set_initial_time_step)
  //       .def("setThreshold", &explicit_RK4::set_threshold)
  //       .def("setFinalTime", &explicit_RK4::set_final_time)
  //       .def("setNumRigidBody", &explicit_RK4::set_num_rigid_body)
  //       .def("initialize", &explicit_RK4::init)
  //       .def("setVelocityFunc",
  //       &explicit_RK4::set_python_velocity_update_func) .def("setXLim",
  //       &explicit_RK4::set_xlim) .def("setYLim", &explicit_RK4::set_ylim)
  //       .def("setZLim", &explicit_RK4::set_zlim)
  //       .def("run", &explicit_RK4::run);
}
