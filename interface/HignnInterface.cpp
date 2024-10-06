#include "HignnModel.hpp"
#include "TimeIntegrator.hpp"
#include "PotentialForce.hpp"
#include "NeighborLists.hpp"

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
      .def("set_eta", &HignnModel::SetEta)
      .def("set_max_iter", &HignnModel::SetMaxIter)
      .def("set_mat_pool_size_factor", &HignnModel::SetMatPoolSizeFactor)
      .def("set_max_far_dot_work_node_size",
           &HignnModel::SetMaxFarDotWorkNodeSize)
      .def("set_max_relative_coord", &HignnModel::SetMaxRelativeCoord)
      .def("set_post_check_flag", &HignnModel::SetPostCheckFlag)
      .def("set_use_symmetry_flag", &HignnModel::SetUseSymmetryFlag)
      .def("set_max_far_field_distance", &HignnModel::SetMaxFarFieldDistance)
      .def("update_coord", &HignnModel::UpdateCoord)
      .def("dot", &HignnModel::Dot)
      .def("dense_dot", pybind11::overload_cast<pybind11::array_t<float> &,
                                                pybind11::array_t<float> &>(
                            &HignnModel::DenseDot));

  py::class_<ExplicitEuler>(m, "ExplicitEuler")
      .def(py::init())
      .def("set_time_step", &ExplicitEuler::SetTimeStep)
      .def("set_final_time", &ExplicitEuler::SetFinalTime)
      .def("set_num_rigid_body", &ExplicitEuler::SetNumRigidBody)
      .def("set_velocity_func", &ExplicitEuler::SetVelocityUpdateFunc)
      .def("set_x_lim", &ExplicitEuler::SetXLim)
      .def("set_y_lim", &ExplicitEuler::SetYLim)
      .def("set_z_lim", &ExplicitEuler::SetZLim)
      .def("set_output_step", &ExplicitEuler::SetOutputStep)
      .def("initialize", &ExplicitEuler::Init)
      .def("run", &ExplicitEuler::Run);

  py::class_<ExplicitRk4>(m, "ExplicitRk4")
      .def(py::init())
      .def("set_time_step", &ExplicitRk4::SetTimeStep)
      .def("set_threshold", &ExplicitRk4::SetThreshold)
      .def("set_final_time", &ExplicitRk4::SetFinalTime)
      .def("set_num_rigid_body", &ExplicitRk4::SetNumRigidBody)
      .def("set_velocity_func", &ExplicitRk4::SetVelocityUpdateFunc)
      .def("set_x_lim", &ExplicitRk4::SetXLim)
      .def("set_y_lim", &ExplicitRk4::SetYLim)
      .def("set_z_lim", &ExplicitRk4::SetZLim)
      .def("set_output_step", &ExplicitEuler::SetOutputStep)
      .def("initialize", &ExplicitRk4::Init)
      .def("run", &ExplicitRk4::Run);

  py::class_<PotentialForce>(m, "PotentialForce")
      .def(py::init())
      .def("set_periodic", &PotentialForce::SetPeriodic)
      .def("set_domain", &PotentialForce::SetDomain)
      .def("set_two_body_epsilon", &PotentialForce::SetTwoBodyEpsilon)
      .def("get_potential_force", &PotentialForce::GetPotentialForce);

  py::class_<NeighborLists>(m, "NeighborLists")
      .def(py::init())
      .def("set_periodic", &NeighborLists::SetPeriodic)
      .def("set_domain", &NeighborLists::SetDomain)
      .def("update_coord", &NeighborLists::UpdateCoord)
      .def("set_two_body_epsilon", &NeighborLists::SetTwoBodyEpsilon)
      .def("set_three_body_epsilon", &NeighborLists::SetThreeBodyEpsilon)
      .def("get_two_body_edge_info", &NeighborLists::GetTwoBodyEdgeInfo)
      .def("get_two_body_edge_info_by_index",
           &NeighborLists::GetTwoBodyEdgeInfoByIndex)
      .def("get_three_body_edge_info", &NeighborLists::GetThreeBodyEdgeInfo)
      .def("get_three_body_edge_info_by_index",
           &NeighborLists::GetThreeBodyEdgeInfoByIndex)
      .def("get_three_body_edge_self_info",
           &NeighborLists::GetThreeBodyEdgeSelfInfo)
      .def("get_three_body_edge_self_info_by_index",
           &NeighborLists::GetThreeBodyEdgeSelfInfoByIndex)
      .def("build_two_body_info", &NeighborLists::BuildTwoBodyInfo)
      .def("build_three_body_info", &NeighborLists::BuildThreeBodyInfo);
}