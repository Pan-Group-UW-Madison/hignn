#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <time_integrator.hpp>

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(pyTimeIntegrator, m) {
  m.doc() = R"pbdoc(
        Time Integrator for Python
    )pbdoc";

  py::class_<explicit_RK4>(m, "explicitRK4")
      .def(py::init())
      .def("setTimeStep", &explicit_RK4::set_initial_time_step)
      .def("setThreshold", &explicit_RK4::set_threshold)
      .def("setFinalTime", &explicit_RK4::set_final_time)
      .def("setNumRigidBody", &explicit_RK4::set_num_rigid_body)
      .def("initialize", &explicit_RK4::init)
      .def("setVelocityFunc", &explicit_RK4::set_python_velocity_update_func)
      .def("setXLim", &explicit_RK4::set_xlim)
      .def("setYLim", &explicit_RK4::set_ylim)
      .def("setZLim", &explicit_RK4::set_zlim)
      .def("run", &explicit_RK4::run);

  py::class_<explicit_euler>(m, "explicitEuler")
      .def(py::init())
      .def("setTimeStep", &explicit_euler::set_time_step)
      .def("setFinalTime", &explicit_euler::set_final_time)
      .def("setNumRigidBody", &explicit_euler::set_num_rigid_body)
      .def("setOutputStep", &explicit_euler::set_output_step)
      .def("initialize", &explicit_euler::init)
      .def("setVelocityFunc", &explicit_euler::set_python_velocity_update_func)
      .def("setXLim", &explicit_euler::set_xlim)
      .def("setYLim", &explicit_euler::set_ylim)
      .def("setZLim", &explicit_euler::set_zlim)
      .def("run", &explicit_euler::run);
}