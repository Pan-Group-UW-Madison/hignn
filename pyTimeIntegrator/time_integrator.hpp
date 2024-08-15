#ifndef TIME_INTEGRATOR_HPP
#define TIME_INTEGRATOR_HPP

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// #include <Kokkos_Core.hpp>

#include <quaternion.hpp>
#include <vec3.hpp>

#include <fstream>
#include <string>
#include <vector>

class explicit_euler {
private:
  float time_step;
  float final_time;

  std::string output_filename;

  int func_count;
  int output_step;
  std::size_t num_rigid_body;

  std::vector<vec3> position0;
  std::vector<quaternion> orientation0;

  std::vector<vec3> position;
  std::vector<quaternion> orientation;

  std::vector<vec3> velocity;
  std::vector<vec3> angular_velocity;

  vec3 domain_limit[2];
  std::vector<vec3> position_offset;

  std::function<pybind11::array_t<float>(float, pybind11::array_t<float>)>
      python_velocity_update;

  void velocity_update(const float t, const std::vector<vec3> &position,
                       const std::vector<quaternion> &orientation,
                       std::vector<vec3> &velocity,
                       std::vector<vec3> &angular_velocity) {
    // convert
    pybind11::array_t<float> input;
    input.resize({(int)num_rigid_body, 6});

    auto input_data = input.mutable_unchecked<2>();
    // deep copy
#pragma omp parallel for schedule(static)
    for (std::size_t num = 0; num < num_rigid_body; num++) {
      for (int j = 0; j < 3; j++) {
        // periodic boundary condition
        if (position[num][j] > domain_limit[1][j]) {
          position_offset[num][j] = domain_limit[0][j] - domain_limit[1][j];
        } else if (position[num][j] < domain_limit[0][j]) {
          position_offset[num][j] = domain_limit[1][j] - domain_limit[0][j];
        } else {
          position_offset[num][j] = 0.0;
        }
        input_data(num, j) = position[num][j] + position_offset[num][j];
      }
      float row, pitch, yaw;
      orientation[num].to_euler_angles(row, pitch, yaw);
      input_data(num, 3) = row;
      input_data(num, 4) = pitch;
      input_data(num, 5) = yaw;
    }

    auto result = python_velocity_update(t, input);

    auto result_data = result.unchecked<2>();
    // deep copy
#pragma omp parallel for schedule(static)
    for (std::size_t num = 0; num < num_rigid_body; num++) {
      for (int j = 0; j < 3; j++) {
        velocity[num][j] = result_data(num, j);
        angular_velocity[num][j] = result_data(num, j + 3);
      }
    }
  }

public:
  explicit_euler()
      : time_step(1.0), final_time(10.0), output_filename("output.txt"),
        output_step(1) {
    domain_limit[0][0] = -1;
    domain_limit[1][0] = 1;
    domain_limit[0][1] = -1;
    domain_limit[1][1] = 1;
    domain_limit[0][2] = -1;
    domain_limit[1][2] = 1;
  }

  void set_time_step(float _time_step) { time_step = _time_step; }

  void set_final_time(float _final_time) { final_time = _final_time; }

  void set_num_rigid_body(std::size_t _num_rigid_body) {
    num_rigid_body = _num_rigid_body;

    position0.resize(num_rigid_body);
    orientation0.resize(num_rigid_body);

    position.resize(num_rigid_body);
    orientation.resize(num_rigid_body);

    velocity.resize(num_rigid_body);
    angular_velocity.resize(num_rigid_body);

    position_offset.resize(num_rigid_body);
  }

  void set_python_velocity_update_func(
      std::function<pybind11::array_t<float>(float, pybind11::array_t<float>)>
          &func) {
    python_velocity_update = func;
  }

  void set_xlim(pybind11::list xlim) {
    domain_limit[0][0] = pybind11::cast<float>(xlim[0]);
    domain_limit[1][0] = pybind11::cast<float>(xlim[1]);
  }

  void set_ylim(pybind11::list ylim) {
    domain_limit[0][1] = pybind11::cast<float>(ylim[0]);
    domain_limit[1][1] = pybind11::cast<float>(ylim[1]);
  }

  void set_zlim(pybind11::list zlim) {
    domain_limit[0][2] = pybind11::cast<float>(zlim[0]);
    domain_limit[1][2] = pybind11::cast<float>(zlim[1]);
  }

  void set_output_step(int _output_step) { output_step = _output_step; }

  void init(pybind11::array_t<float> init_position) {
    pybind11::buffer_info buf = init_position.request();

    if (buf.ndim != 2) {
      throw std::runtime_error(
          "Number of dimensions of input positions must be two");
    }

    if ((std::size_t)init_position.shape(0) != num_rigid_body) {
      throw std::runtime_error("Inconsistent number of rigid bodys");
    }

    auto data = init_position.unchecked<2>();

    // deep copy
#pragma omp parallel for schedule(static)
    for (std::size_t num = 0; num < num_rigid_body; num++) {
      for (int j = 0; j < 3; j++)
        position0[num][j] = data(num, j);
      orientation0[num] = quaternion(data(num, 3), data(num, 4), data(num, 5));

      position[num] = position0[num];
      orientation[num] = orientation0[num];
    }
  }

  int get_func_count() { return func_count; }

  void run() {
    func_count = 0;
    float t = 0;
    float h0 = time_step;

    std::ofstream output_file(output_filename);
    output_file << 0 << '\t';
    for (std::size_t num = 0; num < num_rigid_body; num++) {
      for (int i = 0; i < 3; i++)
        output_file << position[num][i] << '\t';
      float roll, pitch, yaw;
      orientation[num].to_euler_angles(roll, pitch, yaw);
      output_file << roll << '\t' << pitch << '\t' << yaw << '\t';
    }
    output_file << std::endl;

    while (t < final_time - 1e-5) {
      func_count++;
      float dt = std::min(h0, final_time - t);

      velocity_update(t + dt, position, orientation, velocity,
                      angular_velocity);

#pragma omp parallel for schedule(static)
      for (std::size_t num = 0; num < num_rigid_body; num++) {
        position[num] = position[num] + velocity[num] * dt;
        orientation[num].Cross(orientation[num],
                               quaternion(angular_velocity[num], dt));

        for (int j = 0; j < 3; j++) {
          // periodic boundary condition
          if (position[num][j] > domain_limit[1][j]) {
            position_offset[num][j] = domain_limit[0][j] - domain_limit[1][j];
          } else if (position[num][j] < domain_limit[0][j]) {
            position_offset[num][j] = domain_limit[1][j] - domain_limit[0][j];
          } else {
            position_offset[num][j] = 0.0;
          }
        }
      }

      t += dt;

      if (func_count % output_step == 0) {
        output_file << t << '\t';
        for (std::size_t num = 0; num < num_rigid_body; num++) {
          for (int i = 0; i < 3; i++)
            output_file << position[num][i] + position_offset[num][i] << '\t';
          float roll, pitch, yaw;
          orientation[num].to_euler_angles(roll, pitch, yaw);
          output_file << roll << '\t' << pitch << '\t' << yaw << '\t';
        }
        output_file << std::endl;
      }

      for (std::size_t num = 0; num < num_rigid_body; num++) {
        position[num] = position[num] + position_offset[num];
      }
    }
  }
};

class explicit_RK4 {
private:
  // constants for integration
  const float a21 = static_cast<float>(1) / static_cast<float>(5);

  const float a31 = static_cast<float>(3) / static_cast<float>(40);
  const float a32 = static_cast<float>(9) / static_cast<float>(40);

  const float a41 = static_cast<float>(44) / static_cast<float>(45);
  const float a42 = static_cast<float>(-56) / static_cast<float>(15);
  const float a43 = static_cast<float>(32) / static_cast<float>(9);

  const float a51 = static_cast<float>(19372) / static_cast<float>(6561);
  const float a52 = static_cast<float>(-25360) / static_cast<float>(2187);
  const float a53 = static_cast<float>(64448) / static_cast<float>(6561);
  const float a54 = static_cast<float>(-212) / static_cast<float>(729);

  const float a61 = static_cast<float>(9017) / static_cast<float>(3168);
  const float a62 = static_cast<float>(-355) / static_cast<float>(33);
  const float a63 = static_cast<float>(46732) / static_cast<float>(5247);
  const float a64 = static_cast<float>(49) / static_cast<float>(176);
  const float a65 = static_cast<float>(-5103) / static_cast<float>(18656);

  const float b1 = static_cast<float>(35) / static_cast<float>(384);
  const float b3 = static_cast<float>(500) / static_cast<float>(1113);
  const float b4 = static_cast<float>(125) / static_cast<float>(192);
  const float b5 = static_cast<float>(-2187) / static_cast<float>(6784);
  const float b6 = static_cast<float>(11) / static_cast<float>(84);

  const float dc1 = b1 - static_cast<float>(5179) / static_cast<float>(57600);
  const float dc3 = b3 - static_cast<float>(7571) / static_cast<float>(16695);
  const float dc4 = b4 - static_cast<float>(393) / static_cast<float>(640);
  const float dc5 =
      b5 - static_cast<float>(-92097) / static_cast<float>(339200);
  const float dc6 = b6 - static_cast<float>(187) / static_cast<float>(2100);
  const float dc7 = static_cast<float>(-1) / static_cast<float>(40);

  const float c[7] = {static_cast<float>(0),
                      static_cast<float>(1) / static_cast<float>(5),
                      static_cast<float>(3) / static_cast<float>(10),
                      static_cast<float>(4) / static_cast<float>(5),
                      static_cast<float>(8) / static_cast<float>(9),
                      static_cast<float>(1),
                      static_cast<float>(1)};

  float time_step;
  float threshold;
  float final_time;

  std::string output_filename;

  int func_count;

  std::size_t num_rigid_body;

  std::vector<vec3> position0;
  std::vector<quaternion> orientation0;

  std::vector<vec3> position;
  std::vector<quaternion> orientation;

  std::vector<vec3> velocity;
  std::vector<vec3> angular_velocity;

  std::vector<vec3> velocity_k1;
  std::vector<vec3> velocity_k2;
  std::vector<vec3> velocity_k3;
  std::vector<vec3> velocity_k4;
  std::vector<vec3> velocity_k5;
  std::vector<vec3> velocity_k6;
  std::vector<vec3> velocity_k7;

  std::vector<vec3> angular_velocity_k1;
  std::vector<vec3> angular_velocity_k2;
  std::vector<vec3> angular_velocity_k3;
  std::vector<vec3> angular_velocity_k4;
  std::vector<vec3> angular_velocity_k5;
  std::vector<vec3> angular_velocity_k6;
  std::vector<vec3> angular_velocity_k7;

  std::vector<quaternion> intermediate_quaternion1;
  std::vector<quaternion> intermediate_quaternion2;

  vec3 domain_limit[2];
  std::vector<vec3> position_offset;

  std::function<pybind11::array_t<float>(float, pybind11::array_t<float>)>
      python_velocity_update;

  void velocity_update(const float t, const std::vector<vec3> &position,
                       const std::vector<quaternion> &orientation,
                       std::vector<vec3> &velocity,
                       std::vector<vec3> &angular_velocity) {
    // convert
    pybind11::array_t<float> input;
    input.resize({(int)num_rigid_body, 6});

    auto input_data = input.mutable_unchecked<2>();
    // deep copy
#pragma omp parallel for schedule(static)
    for (std::size_t num = 0; num < num_rigid_body; num++) {
      for (int j = 0; j < 3; j++) {
        // periodic boundary condition
        if (position[num][j] > domain_limit[1][j]) {
          position_offset[num][j] = domain_limit[0][j] - domain_limit[1][j];
        } else if (position[num][j] < domain_limit[0][j]) {
          position_offset[num][j] = domain_limit[1][j] - domain_limit[0][j];
        } else {
          position_offset[num][j] = 0.0;
        }
        input_data(num, j) = position[num][j] + position_offset[num][j];
      }
      float row, pitch, yaw;
      orientation[num].to_euler_angles(row, pitch, yaw);
      input_data(num, 3) = row;
      input_data(num, 4) = pitch;
      input_data(num, 5) = yaw;
    }

    auto result = python_velocity_update(t, input);

    auto result_data = result.unchecked<2>();
    // deep copy
#pragma omp parallel for schedule(static)
    for (std::size_t num = 0; num < num_rigid_body; num++) {
      for (int j = 0; j < 3; j++) {
        velocity[num][j] = result_data(num, j);
        angular_velocity[num][j] = result_data(num, j + 3);
      }
    }
  }

public:
  explicit_RK4()
      : time_step(1.0), threshold(1e-3), final_time(10.0),
        output_filename("output.txt") {
    domain_limit[0][0] = -1;
    domain_limit[1][0] = 1;
    domain_limit[0][1] = -1;
    domain_limit[1][1] = 1;
    domain_limit[0][2] = -1;
    domain_limit[1][2] = 1;
  }

  void set_initial_time_step(float _time_step) { time_step = _time_step; }

  void set_threshold(float _threshold) { threshold = _threshold; }

  void set_final_time(float _final_time) { final_time = _final_time; }

  void set_num_rigid_body(std::size_t _num_rigid_body) {
    num_rigid_body = _num_rigid_body;

    position0.resize(num_rigid_body);
    orientation0.resize(num_rigid_body);

    position.resize(num_rigid_body);
    orientation.resize(num_rigid_body);

    velocity.resize(num_rigid_body);
    angular_velocity.resize(num_rigid_body);

    velocity_k1.resize(num_rigid_body);
    velocity_k2.resize(num_rigid_body);
    velocity_k3.resize(num_rigid_body);
    velocity_k4.resize(num_rigid_body);
    velocity_k5.resize(num_rigid_body);
    velocity_k6.resize(num_rigid_body);
    velocity_k7.resize(num_rigid_body);

    angular_velocity_k1.resize(num_rigid_body);
    angular_velocity_k2.resize(num_rigid_body);
    angular_velocity_k3.resize(num_rigid_body);
    angular_velocity_k4.resize(num_rigid_body);
    angular_velocity_k5.resize(num_rigid_body);
    angular_velocity_k6.resize(num_rigid_body);
    angular_velocity_k7.resize(num_rigid_body);

    position_offset.resize(num_rigid_body);
  }

  void set_python_velocity_update_func(
      std::function<pybind11::array_t<float>(float, pybind11::array_t<float>)>
          &func) {
    python_velocity_update = func;
  }

  void set_xlim(pybind11::list xlim) {
    domain_limit[0][0] = pybind11::cast<float>(xlim[0]);
    domain_limit[1][0] = pybind11::cast<float>(xlim[1]);
  }

  void set_ylim(pybind11::list ylim) {
    domain_limit[0][1] = pybind11::cast<float>(ylim[0]);
    domain_limit[1][1] = pybind11::cast<float>(ylim[1]);
  }

  void set_zlim(pybind11::list zlim) {
    domain_limit[0][2] = pybind11::cast<float>(zlim[0]);
    domain_limit[1][2] = pybind11::cast<float>(zlim[1]);
  }

  void init(pybind11::array_t<float> init_position) {
    pybind11::buffer_info buf = init_position.request();

    if (buf.ndim != 2) {
      throw std::runtime_error(
          "Number of dimensions of input positions must be two");
    }

    if ((std::size_t)init_position.shape(0) != num_rigid_body) {
      throw std::runtime_error("Inconsistent number of rigid bodys");
    }

    auto data = init_position.unchecked<2>();

    // deep copy
#pragma omp parallel for schedule(static)
    for (std::size_t num = 0; num < num_rigid_body; num++) {
      for (int j = 0; j < 3; j++)
        position0[num][j] = data(num, j);
      orientation0[num] = quaternion(data(num, 3), data(num, 4), data(num, 5));

      position[num] = position0[num];
      orientation[num] = orientation0[num];
    }
  }

  int get_func_count() { return func_count; }

  void run() {
    func_count = 0;
    float t = 0;
    float h0 = time_step;

    std::ofstream output_file(output_filename);
    output_file << 0 << '\t';
    for (std::size_t num = 0; num < num_rigid_body; num++) {
      for (int i = 0; i < 3; i++)
        output_file << position[num][i] << '\t';
      float roll, pitch, yaw;
      orientation[num].to_euler_angles(roll, pitch, yaw);
      output_file << roll << '\t' << pitch << '\t' << yaw << '\t';
    }
    output_file << std::endl;

    velocity_update(t, position, orientation, velocity, angular_velocity);

    while (t < final_time - 1e-5) {
      float dt = std::min(h0, final_time - t);

#pragma omp parallel for schedule(static)
      for (std::size_t num = 0; num < num_rigid_body; num++) {
        position0[num] = position[num] + position_offset[num];
        orientation0[num] = orientation[num];
      }

      float err = 100;
      while (err > threshold) {
        func_count++;
        for (int i = 1; i < 7; i++) {
          switch (i) {
          case 1:
#pragma omp parallel for schedule(static)
            for (std::size_t num = 0; num < num_rigid_body; num++) {
              position[num] = position0[num] + velocity_k1[num] * (a21 * dt);
              orientation[num].Cross(
                  orientation0[num],
                  quaternion(angular_velocity_k1[num] * a21, dt));
            }
            break;
          case 2:
#pragma omp parallel for schedule(static)
            for (std::size_t num = 0; num < num_rigid_body; num++) {
              position[num] = position0[num] + velocity_k1[num] * (a31 * dt) +
                              velocity_k2[num] * (a32 * dt);
              orientation[num].Cross(
                  orientation0[num],
                  quaternion((angular_velocity_k1[num] * a31 +
                              angular_velocity_k2[num] * a32),
                             dt));
            }
            break;
          case 3:
#pragma omp parallel for schedule(static)
            for (std::size_t num = 0; num < num_rigid_body; num++) {
              position[num] = position0[num] + velocity_k1[num] * (a41 * dt) +
                              velocity_k2[num] * (a42 * dt) +
                              velocity_k3[num] * (a43 * dt);
              orientation[num].Cross(
                  orientation0[num],
                  quaternion((angular_velocity_k1[num] * a41 +
                              angular_velocity_k2[num] * a42 +
                              angular_velocity_k3[num] * a43),
                             dt));
            }
            break;
          case 4:
#pragma omp parallel for schedule(static)
            for (std::size_t num = 0; num < num_rigid_body; num++) {
              position[num] = position0[num] + velocity_k1[num] * (a51 * dt) +
                              velocity_k2[num] * (a52 * dt) +
                              velocity_k3[num] * (a53 * dt) +
                              velocity_k4[num] * (a54 * dt);
              orientation[num].Cross(
                  orientation0[num],
                  quaternion((angular_velocity_k1[num] * a51 +
                              angular_velocity_k2[num] * a52 +
                              angular_velocity_k3[num] * a53 +
                              angular_velocity_k4[num] * a54),
                             dt));
            }
            break;
          case 5:
#pragma omp parallel for schedule(static)
            for (std::size_t num = 0; num < num_rigid_body; num++) {
              position[num] = position0[num] + velocity_k1[num] * (a61 * dt) +
                              velocity_k2[num] * (a62 * dt) +
                              velocity_k3[num] * (a63 * dt) +
                              velocity_k4[num] * (a64 * dt) +
                              velocity_k5[num] * (a65 * dt);
              orientation[num].Cross(
                  orientation0[num],
                  quaternion((angular_velocity_k1[num] * a61 +
                              angular_velocity_k2[num] * a62 +
                              angular_velocity_k3[num] * a63 +
                              angular_velocity_k4[num] * a64 +
                              angular_velocity_k5[num] * a65),
                             dt));
            }
            break;
          case 6:
#pragma omp parallel for schedule(static)
            for (std::size_t num = 0; num < num_rigid_body; num++) {
              position[num] =
                  position0[num] + velocity_k1[num] * (b1 * dt) +
                  velocity_k3[num] * (b3 * dt) + velocity_k4[num] * (b4 * dt) +
                  velocity_k5[num] * (b5 * dt) + velocity_k6[num] * (b6 * dt);
              orientation[num].Cross(orientation0[num],
                                     quaternion((angular_velocity_k1[num] * b1 +
                                                 angular_velocity_k3[num] * b3 +
                                                 angular_velocity_k4[num] * b4 +
                                                 angular_velocity_k5[num] * b5 +
                                                 angular_velocity_k6[num] * b6),
                                                dt));
            }
            break;
          }

          velocity_update(t + c[i] * dt, position, orientation, velocity,
                          angular_velocity);

          // update velocity
          switch (i) {
          case 1:
#pragma omp parallel for schedule(static)
            for (std::size_t num = 0; num < num_rigid_body; num++) {
              velocity_k2[num] = velocity[num];
              angular_velocity_k2[num] = dexpinv(
                  angular_velocity_k1[num] * a21 * dt, angular_velocity[num]);
            }
            break;
          case 2:
#pragma omp parallel for schedule(static)
            for (std::size_t num = 0; num < num_rigid_body; num++) {
              velocity_k3[num] = velocity[num];
              angular_velocity_k3[num] =
                  dexpinv((angular_velocity_k1[num] * a31 +
                           angular_velocity_k2[num] * a32) *
                              dt,
                          angular_velocity[num]);
            }
            break;
          case 3:
#pragma omp parallel for schedule(static)
            for (std::size_t num = 0; num < num_rigid_body; num++) {
              velocity_k4[num] = velocity[num];
              angular_velocity_k4[num] =
                  dexpinv((angular_velocity_k1[num] * a41 +
                           angular_velocity_k2[num] * a42 +
                           angular_velocity_k3[num] * a43) *
                              dt,
                          angular_velocity[num]);
            }
            break;
          case 4:
#pragma omp parallel for schedule(static)
            for (std::size_t num = 0; num < num_rigid_body; num++) {
              velocity_k5[num] = velocity[num];
              angular_velocity_k5[num] =
                  dexpinv((angular_velocity_k1[num] * a51 +
                           angular_velocity_k2[num] * a52 +
                           angular_velocity_k3[num] * a53 +
                           angular_velocity_k4[num] * a54) *
                              dt,
                          angular_velocity[num]);
            }
            break;
          case 5:
#pragma omp parallel for schedule(static)
            for (std::size_t num = 0; num < num_rigid_body; num++) {
              velocity_k6[num] = velocity[num];
              angular_velocity_k6[num] =
                  dexpinv((angular_velocity_k1[num] * a61 +
                           angular_velocity_k2[num] * a62 +
                           angular_velocity_k3[num] * a63 +
                           angular_velocity_k4[num] * a64 +
                           angular_velocity_k5[num] * a65) *
                              dt,
                          angular_velocity[num]);
            }
            break;
          case 6:
#pragma omp parallel for schedule(static)
            for (std::size_t num = 0; num < num_rigid_body; num++) {
              velocity_k7[num] = velocity[num];
              angular_velocity_k7[num] =
                  dexpinv((angular_velocity_k1[num] * b1 +
                           angular_velocity_k3[num] * b3 +
                           angular_velocity_k4[num] * b4 +
                           angular_velocity_k5[num] * b5 +
                           angular_velocity_k6[num] * b6) *
                              dt,
                          angular_velocity[num]);
            }
            break;
          }
        }

        // error check
        {
          err = 0.0;

#pragma omp parallel for schedule(static) reduction(+ : err)
          for (std::size_t num = 0; num < num_rigid_body; num++) {
            vec3 velocity_err =
                (velocity_k1[num] * dc1 + velocity_k3[num] * dc3 +
                 velocity_k4[num] * dc4 + velocity_k5[num] * dc5 +
                 velocity_k6[num] * dc6 + velocity_k7[num] * dc7) *
                dt;

            err += velocity_err.sqrmag();
          }
          err = sqrt(err);

          if (err > threshold) {
            // reduce time step
            dt = dt * std::max(0.8 * pow(err / threshold, -0.2), 0.1);
            dt = std::max(dt, 1e-6f);
          }
        }
      }

      t += dt;

      // increase time step
      {
        float temp = 1.25 * pow(err / threshold, 0.2);
        if (temp > 0.2) {
          dt = dt / temp;
        } else {
          dt *= 5.0;
        }
      }
      dt = std::min(dt, h0);

      // reset k1 for next step
#pragma omp parallel for schedule(static)
      for (std::size_t num = 0; num < num_rigid_body; num++) {
        velocity_k1[num] = velocity_k7[num];
        angular_velocity_k1[num] = angular_velocity_k7[num];
      }

      output_file << t << '\t';
      for (std::size_t num = 0; num < num_rigid_body; num++) {
        for (int i = 0; i < 3; i++)
          output_file << position[num][i] + position_offset[num][i] << '\t';
        float roll, pitch, yaw;
        orientation[num].to_euler_angles(roll, pitch, yaw);
        output_file << roll << '\t' << pitch << '\t' << yaw << '\t';
      }
      output_file << std::endl;
    }

    // final output
    output_file << t << '\t';
    for (std::size_t num = 0; num < num_rigid_body; num++) {
      for (int i = 0; i < 3; i++)
        output_file << position[num][i] + position_offset[num][i] << '\t';
      float roll, pitch, yaw;
      orientation[num].to_euler_angles(roll, pitch, yaw);
      output_file << roll << '\t' << pitch << '\t' << yaw << '\t';
    }
    output_file << std::endl;
    output_file.close();
  }
};

#endif