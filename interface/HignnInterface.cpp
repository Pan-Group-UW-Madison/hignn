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

  py::class_<HignnModel>(m, "HignnModel")
      .def(py::init<pybind11::array_t<float> &, int>())
      .def("load_two_body_model", &HignnModel::LoadTwoBodyModel)
      .def("load_three_body_model", &HignnModel::LoadThreeBodyModel)
      .def("set_epsilon", &HignnModel::SetEpsilon)
      .def("set_max_iter", &HignnModel::SetMaxIter)
      .def("set_mat_pool_size_factor", &HignnModel::SetMatPoolSizeFactor)
      .def("set_max_far_dot_work_node_size",
           &HignnModel::SetMaxFarDotWorkNodeSize)
      .def("set_max_relative_coord", &HignnModel::SetMaxRelativeCoord)
      .def("set_post_check_flag", &HignnModel::SetPostCheckFlag)
      .def("set_use_symmetry_flag", &HignnModel::SetUseSymmetryFlag)
      .def("set_max_far_field_distance", &HignnModel::SetMaxFarFieldDistance)
      .def("update_coord", &HignnModel::UpdateCoord)
      .def("dot", &HignnModel::Dot);

  py::class_<ExplicitEuler>(m, "ExplicitEuler")
      .def(py::init())
      .def("set_time_step", &ExplicitEuler::SetTimeStep)
      .def("set_final_time", &ExplicitEuler::SetFinalTime)
      .def("set_num_rigid_body", &ExplicitEuler::SetNumRigidBody)
      .def("set_velocity_func", &ExplicitEuler::SetPythonVelocityUpdateFunc)
      .def("set_x_lim", &ExplicitEuler::SetXLim)
      .def("set_y_lim", &ExplicitEuler::SetYLim)
      .def("set_z_lim", &ExplicitEuler::SetZLim)
      .def("set_output_step", &ExplicitEuler::SetOutputStep)
      .def("initialize", &ExplicitEuler::init)
      .def("run", &ExplicitEuler::Run);

  py::class_<ExplicitRk4>(m, "ExplicitRk4")
      .def(py::init())
      .def("set_time_step", &ExplicitRk4::SetInitialTimeStep);
  //   .def("set_threshold", &ExplicitRk4::SetThreshold)
  //   .def("set_finalTime", &ExplicitRk4::SetFinalTime)
  //   .def("set_num_rigid_body", &ExplicitRk4::SetNumRigidBody)
  //   .def("set_velocity_func", &ExplicitRk4::SetPythonVelocityUpdateFunc)
  //   .def("set_x_lim", &ExplicitEuler::SetXlim)
  //   .def("set_y_lim", &ExplicitEuler::SetYlim)
  //   .def("set_z_lim", &ExplicitEuler::SetZlim)
  //   .def("initialize", &ExplicitRk4::init)
  //   .def("run", &ExplicitRk4::run);
}