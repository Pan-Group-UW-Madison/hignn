#include "HignnModel.hpp"
#include "TimeIntegrator.hpp"

template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

PYBIND11_MODULE(hignn, m) {
  m.doc() = R"pbdoc(
        Hydrodynamic Interaction Graph Neural Network
    )pbdoc";

  m.def("Init", &Init, "Initialize MPI and Kokkos");
  m.def("Finalize", &Finalize, "Finalize MPI and Kokkos");

  py::class_<HignnModel>(m, "hignn_model")
      .def(py::init<pybind11::array_t<float> &, int>())
      .def("load_two_body_model", &HignnModel::LoadTwoBodyModel)
      .def("load_three_body_model", &HignnModel::LoadThreeBodyModel)
      .def("set_epsilon", &HignnModel::SetEpsilon)
      .def("set_max_iter", &HignnModel::SetMaxIter)
      .def("set_mat_pool_size_factor", &HignnModel::SetMatPoolSizeFactor)
      .def("set_max_far_dot_work_node_size", &HignnModel::SetMaxFarDotWorkNodeSize)
      .def("set_max_relative_coord", &HignnModel::SetMaxRelativeCoord)
      .def("set_post_check_flag", &HignnModel::SetPostCheckFlag)
      .def("set_use_symmetry_flag", &HignnModel::SetUseSymmetryFlag)
      .def("set_max_far_field_distance", &HignnModel::SetMaxFarFieldDistance)
      .def("update_coord", &HignnModel::UpdateCoord)
      .def("dot", &HignnModel::Dot);

  // Making Py class by myself

  py::class_<ExplicitEuler>(m, "explicit_euler")
            .def(py::init())
            .def("set_time_step", &ExplicitEuler::SetTimeStep)
            .def("set_final_time", &ExplicitEuler::SetFinalTime)
            .def("set_num_rigid_body", &ExplicitEuler::SetNumRigidBody)
            .def("set_velocity_func", &ExplicitEuler::SetPythonVelocityUpdateFunc) 
            .def("set_x_lim", &ExplicitEuler::SetXlim) 
            .def("set_y_lim", &ExplicitEuler::SetYlim)
            .def("set_z_lim", &ExplicitEuler::SetZlim)
            .def("set_output_step", &ExplicitEuler::SetOutputStep)
            .def("initialize", &ExplicitEuler::init)
            .def("run", &ExplicitEuler::run);

  //   py::class_<neighbor_lists>(m, "BodyEdgeInfo")
  //       .def(py::init())
  //       .def("setTargetSites", overload_cast_<pybind11::array_t<float>>()(
  //                                  &neighbor_lists::set_target_sites))
  //       .def("setPeriodic",
  //       overload_cast_<bool>()(&neighbor_lists::set_periodic))
  //       .def("setDomain", overload_cast_<pybind11::array_t<float>>()(
  //                             &neighbor_lists::set_domain))
  //       .def("setTwoBodyEpsilon", &neighbor_lists::set_two_body_epsilon)
  //       .def("setThreeBodyEpsilon", &neighbor_lists::set_three_body_epsilon)
  //       .def("buildTwoBodyEdgeInfo", &neighbor_lists::build_two_body_info)
  //       .def("buildThreeBodyEdgeInfo",
  //       &neighbor_lists::build_three_body_info) .def("getThreeBodyEdgeInfo",
  //       &neighbor_lists::get_three_body_edge_info)
  //       .def("getThreeBodyEdgeInfoByIndex",
  //            overload_cast_<pybind11::array_t<std::size_t>>()(
  //                &neighbor_lists::get_three_body_edge_info_by_index))
  //       .def("getThreeBodyEdgeSelfInfo",
  //            &neighbor_lists::get_three_body_edge_self_info)
  //       .def("getThreeBodyEdgeSelfInfoByIndex",
  //            overload_cast_<pybind11::array_t<std::size_t>>()(
  //                &neighbor_lists::get_three_body_edge_self_info_by_index))
  //       .def("getTwoBodyEdgeInfo", &neighbor_lists::get_two_body_edge_info)
  //       .def("getTwoBodyEdgeInfoByIndex",
  //            overload_cast_<pybind11::array_t<std::size_t>>()(
  //                &neighbor_lists::get_two_body_edge_info_by_index))
  //       .def("getMorseForce", &neighbor_lists::get_morse_force)
  //       .def("getMorseForceByIndex",
  //            overload_cast_<pybind11::array_t<std::size_t>>()(
  //                &neighbor_lists::get_morse_force_by_index))
  //       .def("getEdgeAttr3", &neighbor_lists::get_edge_attr3)
  //       .def("getEdgeAttrSelf", &neighbor_lists::get_edge_attr_self)
  //       .def("getEdgeAttr", &neighbor_lists::get_edge_attr)
  //       .def("getThreeBodyEdgeAttr",
  //            overload_cast_<pybind11::array_t<float>>()(
  //                &neighbor_lists::get_three_body_edge_attr));

      py::class_<ExplicitRk4>(m, "ExplicitRk4")
          .def(py:init())
          .def("set_time_step", &ExplicitRk4::set_Initial_TimeStep)
          .def("set_threshold", &ExplicitRk4::SetThreshold)
          .def("set_finalTime", &ExplicitRk4::SetFinalTime)
          .def("set_num_rigid_body", &ExplicitRk4::SetNumRigidBody)
          .def("set_velocity_func", &ExplicitRk4::SetPythonVelocityUpdateFunc) 
          .def("set_x_lim", &ExplicitEuler::SetXlim) 
          .def("set_y_lim", &ExplicitEuler::SetYlim)
          .def("set_z_lim", &ExplicitEuler::SetZlim)
          .def("initialize", &ExplicitRk4::init)
          .def("run", &ExplicitRk4::run);


      // py::class_<ThreeBodyEdgeInfo>(m, "ThreeBodyEdgeInfo")
      //       .def(py::init())
      //       .def("setTwoBodyEpsilon", &Three_Body_Edge_Info::set_two_body_epsilon)
      //       .def("setThreeBodyEpsilon", &Three_Body_Edge_Info::set_three_body_epsilon)
      //       .def("buildTwoBodyEdgeInfo", &Three_Body_Edge_Info::build_two_body_info)
      //       .def("setPeriodic", overload_cast_<bool>()(&Three_Body_Edge_Info::set_periodic))
      //       .def("setDomain", overload_cast_<pybind11::array_t<float>>()( &Three_Body_Edge_Info::set_domain));
            


}