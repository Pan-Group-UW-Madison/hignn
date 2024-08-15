#ifndef _TimeIntegrator_Hpp_
#define _TimeIntegrator_Hpp_

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Quaternion.hpp>
#include <Vec3.hpp>

#include <fstream>
#include <string>
#include <vector>
#include <sstream>

#include <mpi.h>

// As one pybind11 object is stored in a shared library, it is necessary to
// control the visibility of the object.
class __attribute__((visibility("default"))) ExplicitEuler {
protected:
  float mTimeStep;
  float mFinalTime;

  std::string mOutputFilePrefix;

  int mFuncCount;
  int mOutputStep;
  std::size_t mNumRigidBody;

  std::vector<Vec3> mPosition0;
  std::vector<Quaternion> mOrientation0;

  std::vector<Vec3> mPosition;
  std::vector<Quaternion> mOrientation;

  std::vector<Vec3> mVelocity;
  std::vector<Vec3> mAngularVelocity;

  Vec3 mDomainLimit[2];
  std::vector<Vec3> mPositionOffset;

  std::function<pybind11::array_t<float>(float, pybind11::array_t<float>)>
      python_velocity_update;

  void velocity_update(
      const float t,
      const std::vector<Vec3> &position,
      [[maybe_unused]] const std::vector<Quaternion> &orientation,
      std::vector<Vec3> &velocity,
      std::vector<Vec3> &angularVelocity) {
    // convert
    pybind11::array_t<float> input;
    input.resize({(int)mNumRigidBody, 6});

    auto input_data = input.mutable_unchecked<2>();
    // deep copy
#pragma omp parallel for schedule(static)
    for (std::size_t num = 0; num < mNumRigidBody; num++) {
      for (int j = 0; j < 3; j++) {
        // periodic boundary condition
        if (position[num][j] > mDomainLimit[1][j]) {
          mPositionOffset[num][j] = mDomainLimit[0][j] - mDomainLimit[1][j];
        } else if (position[num][j] < mDomainLimit[0][j]) {
          mPositionOffset[num][j] = mDomainLimit[1][j] - mDomainLimit[0][j];
        } else {
          mPositionOffset[num][j] = 0.0;
        }
        input_data(num, j) = position[num][j] + mPositionOffset[num][j];
      }
      float row, pitch, yaw;
      mOrientation[num].to_euler_angles(row, pitch, yaw);
      input_data(num, 3) = row;
      input_data(num, 4) = pitch;
      input_data(num, 5) = yaw;
    }

    auto result = python_velocity_update(t, input);

    auto result_data = result.unchecked<2>();
    // deep copy
#pragma omp parallel for schedule(static)
    for (std::size_t num = 0; num < mNumRigidBody; num++) {
      for (int j = 0; j < 3; j++) {
        velocity[num][j] = result_data(num, j);
        angularVelocity[num][j] = result_data(num, j + 3);
      }
    }
  }

public:
  ExplicitEuler()
      : mTimeStep(1.0),
        mFinalTime(10.0),
        mOutputFilePrefix("Result/output"),
        mOutputStep(1) {
    mDomainLimit[0][0] = -1;
    mDomainLimit[1][0] = 1;
    mDomainLimit[0][1] = -1;
    mDomainLimit[1][1] = 1;
    mDomainLimit[0][2] = -1;
    mDomainLimit[1][2] = 1;
  }

  void SetTimeStep(const float timeStep) {
    mTimeStep = timeStep;
  }

  void SetFinalTime(const float finalTime) {
    mFinalTime = finalTime;
  }

  void SetNumRigidBody(const std::size_t numRigidBody) {
    mNumRigidBody = numRigidBody;

    mPosition0.resize(mNumRigidBody);
    mOrientation0.resize(mNumRigidBody);

    mPosition.resize(mNumRigidBody);
    mOrientation.resize(mNumRigidBody);

    mVelocity.resize(mNumRigidBody);
    mAngularVelocity.resize(mNumRigidBody);

    mPositionOffset.resize(mNumRigidBody);
  }

  void SetPythonVelocityUpdateFunc(
      const std::function<
          pybind11::array_t<float>(float, pybind11::array_t<float>)> &func) {
    python_velocity_update = func;
  }

  void SetXLim(pybind11::list xLim) {
    mDomainLimit[0][0] = pybind11::cast<float>(xLim[0]);
    mDomainLimit[1][0] = pybind11::cast<float>(xLim[1]);
  }

  void SetYLim(pybind11::list yLim) {
    mDomainLimit[0][1] = pybind11::cast<float>(yLim[0]);
    mDomainLimit[1][1] = pybind11::cast<float>(yLim[1]);
  }

  void SetZLim(pybind11::list zLim) {
    mDomainLimit[0][2] = pybind11::cast<float>(zLim[0]);
    mDomainLimit[1][2] = pybind11::cast<float>(zLim[1]);
  }

  void SetOutputStep(const int outputStep) {
    mOutputStep = outputStep;
  }

  void Init(pybind11::array_t<float> initPosition) {
    pybind11::buffer_info buf = initPosition.request();

    if (buf.ndim != 2) {
      throw std::runtime_error(
          "Number of dimensions of input positions must be two");
    }

    if ((std::size_t)initPosition.shape(0) != mNumRigidBody) {
      throw std::runtime_error("Inconsistent number of rigid bodies");
    }

    auto data = initPosition.unchecked<2>();

    // deep copy
#pragma omp parallel for schedule(static)
    for (std::size_t num = 0; num < mNumRigidBody; num++) {
      for (int j = 0; j < 3; j++)
        mPosition0[num][j] = data(num, j);
      mOrientation0[num] = Quaternion(data(num, 3), data(num, 4), data(num, 5));

      mPosition[num] = mPosition0[num];
      mOrientation[num] = mOrientation0[num];
    }
  }

  int GetFuncCount() {
    return mFuncCount;
  }

  void Run() {
    mFuncCount = 0;
    float t = 0;
    float h0 = mTimeStep;

    int mpiRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

    if (mpiRank == 0) {
      std::ofstream output_file(mOutputFilePrefix + "_" +
                                std::to_string(mFuncCount) + ".txt");
      std::stringstream outputStream;
      for (std::size_t num = 0; num < mNumRigidBody; num++) {
        for (int i = 0; i < 3; i++)
          outputStream << mPosition[num][i] << '\t';
      }
      outputStream << std::endl;
      output_file << outputStream.str();
      output_file.close();
    }

    while (t < mFinalTime - 1e-5) {
      mFuncCount++;
      float dt = std::min(h0, mFinalTime - t);

      velocity_update(t + dt, mPosition, mOrientation, mVelocity,
                      mAngularVelocity);

#pragma omp parallel for schedule(static)
      for (std::size_t num = 0; num < mNumRigidBody; num++) {
        mPosition[num] = mPosition[num] + mVelocity[num] * dt;
        mOrientation[num].Cross(mOrientation[num],
                                Quaternion(mAngularVelocity[num], dt));

        for (int j = 0; j < 3; j++) {
          // periodic boundary condition
          if (mPosition[num][j] > mDomainLimit[1][j]) {
            mPositionOffset[num][j] = mDomainLimit[0][j] - mDomainLimit[1][j];
          } else if (mPosition[num][j] < mDomainLimit[0][j]) {
            mPositionOffset[num][j] = mDomainLimit[1][j] - mDomainLimit[0][j];
          } else {
            mPositionOffset[num][j] = 0.0;
          }
        }
      }

      t += dt;

      if (mFuncCount % mOutputStep == 0) {
        if (mpiRank == 0) {
          std::ofstream output_file(mOutputFilePrefix + "_" +
                                    std::to_string(mFuncCount) + ".txt");
          std::stringstream outputStream;
          for (std::size_t num = 0; num < mNumRigidBody; num++) {
            for (int i = 0; i < 3; i++)
              outputStream << mPosition[num][i] << '\t';
          }
          outputStream << std::endl;
          output_file << outputStream.str();
          output_file.close();
        }
      }

      for (std::size_t num = 0; num < mNumRigidBody; num++) {
        mPosition[num] = mPosition[num] + mPositionOffset[num];
      }
    }
  }
};

class __attribute__((visibility("default"))) ExplicitRk4 {
protected:
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

  float mTimeStep;
  float threshold;
  float mFinalTime;

  std::string mOutputFilePrefix;

  int mFuncCount;

  std::size_t mNumRigidBody;

  std::vector<Vec3> mPosition0;
  std::vector<Quaternion> mOrientation0;

  std::vector<Vec3> mPosition;
  std::vector<Quaternion> mOrientation;

  std::vector<Vec3> mVelocity;
  std::vector<Vec3> angular_velocity;

  std::vector<Vec3> velocity_k1;
  std::vector<Vec3> velocity_k2;
  std::vector<Vec3> velocity_k3;
  std::vector<Vec3> velocity_k4;
  std::vector<Vec3> velocity_k5;
  std::vector<Vec3> velocity_k6;
  std::vector<Vec3> velocity_k7;

  std::vector<Vec3> angular_velocity_k1;
  std::vector<Vec3> angular_velocity_k2;
  std::vector<Vec3> angular_velocity_k3;
  std::vector<Vec3> angular_velocity_k4;
  std::vector<Vec3> angular_velocity_k5;
  std::vector<Vec3> angular_velocity_k6;
  std::vector<Vec3> angular_velocity_k7;

  std::vector<Quaternion> intermediate_Quaternion1;
  std::vector<Quaternion> intermediate_Quaternion2;

  Vec3 domain_limit[2];
  std::vector<Vec3> position_offset;

  std::function<pybind11::array_t<float>(float, pybind11::array_t<float>)>
      python_velocity_update;

  void velocity_update(
      const float t,
      const std::vector<Vec3> &mPosition,
      [[maybe_unused]] const std::vector<Quaternion> &mOrientation,
      std::vector<Vec3> &velocity,
      std::vector<Vec3> &angularVelocity) {
    // convert
    pybind11::array_t<float> input;
    input.resize({(int)mNumRigidBody, 6});

    auto input_data = input.mutable_unchecked<2>();
    // deep copy
#pragma omp parallel for schedule(static)
    for (std::size_t num = 0; num < mNumRigidBody; num++) {
      for (int j = 0; j < 3; j++) {
        // periodic boundary condition
        if (mPosition[num][j] > domain_limit[1][j]) {
          position_offset[num][j] = domain_limit[0][j] - domain_limit[1][j];
        } else if (mPosition[num][j] < domain_limit[0][j]) {
          position_offset[num][j] = domain_limit[1][j] - domain_limit[0][j];
        } else {
          position_offset[num][j] = 0.0;
        }
        input_data(num, j) = mPosition[num][j] + position_offset[num][j];
      }
      float row, pitch, yaw;
      mOrientation[num].to_euler_angles(row, pitch, yaw);
      input_data(num, 3) = row;
      input_data(num, 4) = pitch;
      input_data(num, 5) = yaw;
    }

    auto result = python_velocity_update(t, input);

    auto result_data = result.unchecked<2>();
    // deep copy
#pragma omp parallel for schedule(static)
    for (std::size_t num = 0; num < mNumRigidBody; num++) {
      for (int j = 0; j < 3; j++) {
        velocity[num][j] = result_data(num, j);
        angularVelocity[num][j] = result_data(num, j + 3);
      }
    }
  }

public:
  ExplicitRk4()
      : mTimeStep(1.0),
        threshold(1e-3),
        mFinalTime(10.0),
        mOutputFilePrefix("result/output") {
    domain_limit[0][0] = -1;
    domain_limit[1][0] = 1;
    domain_limit[0][1] = -1;
    domain_limit[1][1] = 1;
    domain_limit[0][2] = -1;
    domain_limit[1][2] = 1;
  }

  void SetInitialTimeStep(float _mTimeStep) {
    mTimeStep = _mTimeStep;
  }

  void SetThreshold(float _threshold) {
    threshold = _threshold;
  }

  void SetFinalTime(float _mFinalTime) {
    mFinalTime = _mFinalTime;
  }

  void SetNumRigidBody(std::size_t _mNumRigidBody) {
    mNumRigidBody = _mNumRigidBody;

    mPosition0.resize(mNumRigidBody);
    mOrientation0.resize(mNumRigidBody);

    mPosition.resize(mNumRigidBody);
    mOrientation.resize(mNumRigidBody);

    mVelocity.resize(mNumRigidBody);
    angular_velocity.resize(mNumRigidBody);

    velocity_k1.resize(mNumRigidBody);
    velocity_k2.resize(mNumRigidBody);
    velocity_k3.resize(mNumRigidBody);
    velocity_k4.resize(mNumRigidBody);
    velocity_k5.resize(mNumRigidBody);
    velocity_k6.resize(mNumRigidBody);
    velocity_k7.resize(mNumRigidBody);

    angular_velocity_k1.resize(mNumRigidBody);
    angular_velocity_k2.resize(mNumRigidBody);
    angular_velocity_k3.resize(mNumRigidBody);
    angular_velocity_k4.resize(mNumRigidBody);
    angular_velocity_k5.resize(mNumRigidBody);
    angular_velocity_k6.resize(mNumRigidBody);
    angular_velocity_k7.resize(mNumRigidBody);

    position_offset.resize(mNumRigidBody);
  }

  void SetPythonVelocityUpdateFunc(
      const std::function<
          pybind11::array_t<float>(float, pybind11::array_t<float>)> &func) {
    python_velocity_update = func;
  }

  void SetXLim(pybind11::list xLim) {
    domain_limit[0][0] = pybind11::cast<float>(xLim[0]);
    domain_limit[1][0] = pybind11::cast<float>(xLim[1]);
  }

  void SetYLim(pybind11::list yLim) {
    domain_limit[0][1] = pybind11::cast<float>(yLim[0]);
    domain_limit[1][1] = pybind11::cast<float>(yLim[1]);
  }

  void SetZLim(pybind11::list zLim) {
    domain_limit[0][2] = pybind11::cast<float>(zLim[0]);
    domain_limit[1][2] = pybind11::cast<float>(zLim[1]);
  }

  void Init(pybind11::array_t<float> init_position) {
    pybind11::buffer_info buf = init_position.request();

    if (buf.ndim != 2) {
      throw std::runtime_error(
          "Number of dimensions of input positions must be two");
    }

    if ((std::size_t)init_position.shape(0) != mNumRigidBody) {
      throw std::runtime_error("Inconsistent number of rigid bodies");
    }

    auto data = init_position.unchecked<2>();

    // deep copy
#pragma omp parallel for schedule(static)
    for (std::size_t num = 0; num < mNumRigidBody; num++) {
      for (int j = 0; j < 3; j++)
        mPosition0[num][j] = data(num, j);
      mOrientation0[num] = Quaternion(data(num, 3), data(num, 4), data(num, 5));

      mPosition[num] = mPosition0[num];
      mOrientation[num] = mOrientation0[num];
    }
  }

  int GetFuncCount() {
    return mFuncCount;
  }

  void Run() {
    mFuncCount = 0;
    float t = 0;
    float h0 = mTimeStep;

    std::ofstream output_file(mOutputFilePrefix);
    output_file << 0 << '\t';
    for (std::size_t num = 0; num < mNumRigidBody; num++) {
      for (int i = 0; i < 3; i++)
        output_file << mPosition[num][i] << '\t';
      float roll, pitch, yaw;
      mOrientation[num].to_euler_angles(roll, pitch, yaw);
      output_file << roll << '\t' << pitch << '\t' << yaw << '\t';
    }
    output_file << std::endl;

    velocity_update(t, mPosition, mOrientation, mVelocity, angular_velocity);

    while (t < mFinalTime - 1e-5) {
      float dt = std::min(h0, mFinalTime - t);

#pragma omp parallel for schedule(static)
      for (std::size_t num = 0; num < mNumRigidBody; num++) {
        mPosition0[num] = mPosition[num] + position_offset[num];
        mOrientation0[num] = mOrientation[num];
      }

      float err = 100;
      while (err > threshold) {
        mFuncCount++;
        for (int i = 1; i < 7; i++) {
          switch (i) {
            case 1:
#pragma omp parallel for schedule(static)
              for (std::size_t num = 0; num < mNumRigidBody; num++) {
                mPosition[num] =
                    mPosition0[num] + velocity_k1[num] * (a21 * dt);
                mOrientation[num].Cross(
                    mOrientation0[num],
                    Quaternion(angular_velocity_k1[num] * a21, dt));
              }
              break;
            case 2:
#pragma omp parallel for schedule(static)
              for (std::size_t num = 0; num < mNumRigidBody; num++) {
                mPosition[num] = mPosition0[num] +
                                 velocity_k1[num] * (a31 * dt) +
                                 velocity_k2[num] * (a32 * dt);
                mOrientation[num].Cross(
                    mOrientation0[num],
                    Quaternion((angular_velocity_k1[num] * a31 +
                                angular_velocity_k2[num] * a32),
                               dt));
              }
              break;
            case 3:
#pragma omp parallel for schedule(static)
              for (std::size_t num = 0; num < mNumRigidBody; num++) {
                mPosition[num] = mPosition0[num] +
                                 velocity_k1[num] * (a41 * dt) +
                                 velocity_k2[num] * (a42 * dt) +
                                 velocity_k3[num] * (a43 * dt);
                mOrientation[num].Cross(
                    mOrientation0[num],
                    Quaternion((angular_velocity_k1[num] * a41 +
                                angular_velocity_k2[num] * a42 +
                                angular_velocity_k3[num] * a43),
                               dt));
              }
              break;
            case 4:
#pragma omp parallel for schedule(static)
              for (std::size_t num = 0; num < mNumRigidBody; num++) {
                mPosition[num] = mPosition0[num] +
                                 velocity_k1[num] * (a51 * dt) +
                                 velocity_k2[num] * (a52 * dt) +
                                 velocity_k3[num] * (a53 * dt) +
                                 velocity_k4[num] * (a54 * dt);
                mOrientation[num].Cross(
                    mOrientation0[num],
                    Quaternion((angular_velocity_k1[num] * a51 +
                                angular_velocity_k2[num] * a52 +
                                angular_velocity_k3[num] * a53 +
                                angular_velocity_k4[num] * a54),
                               dt));
              }
              break;
            case 5:
#pragma omp parallel for schedule(static)
              for (std::size_t num = 0; num < mNumRigidBody; num++) {
                mPosition[num] = mPosition0[num] +
                                 velocity_k1[num] * (a61 * dt) +
                                 velocity_k2[num] * (a62 * dt) +
                                 velocity_k3[num] * (a63 * dt) +
                                 velocity_k4[num] * (a64 * dt) +
                                 velocity_k5[num] * (a65 * dt);
                mOrientation[num].Cross(
                    mOrientation0[num],
                    Quaternion((angular_velocity_k1[num] * a61 +
                                angular_velocity_k2[num] * a62 +
                                angular_velocity_k3[num] * a63 +
                                angular_velocity_k4[num] * a64 +
                                angular_velocity_k5[num] * a65),
                               dt));
              }
              break;
            case 6:
#pragma omp parallel for schedule(static)
              for (std::size_t num = 0; num < mNumRigidBody; num++) {
                mPosition[num] =
                    mPosition0[num] + velocity_k1[num] * (b1 * dt) +
                    velocity_k3[num] * (b3 * dt) +
                    velocity_k4[num] * (b4 * dt) +
                    velocity_k5[num] * (b5 * dt) + velocity_k6[num] * (b6 * dt);
                mOrientation[num].Cross(
                    mOrientation0[num],
                    Quaternion((angular_velocity_k1[num] * b1 +
                                angular_velocity_k3[num] * b3 +
                                angular_velocity_k4[num] * b4 +
                                angular_velocity_k5[num] * b5 +
                                angular_velocity_k6[num] * b6),
                               dt));
              }
              break;
          }

          velocity_update(t + c[i] * dt, mPosition, mOrientation, mVelocity,
                          angular_velocity);

          // update velocity
          switch (i) {
            case 1:
#pragma omp parallel for schedule(static)
              for (std::size_t num = 0; num < mNumRigidBody; num++) {
                velocity_k2[num] = mVelocity[num];
                angular_velocity_k2[num] = dexpinv(
                    angular_velocity_k1[num] * a21 * dt, angular_velocity[num]);
              }
              break;
            case 2:
#pragma omp parallel for schedule(static)
              for (std::size_t num = 0; num < mNumRigidBody; num++) {
                velocity_k3[num] = mVelocity[num];
                angular_velocity_k3[num] =
                    dexpinv((angular_velocity_k1[num] * a31 +
                             angular_velocity_k2[num] * a32) *
                                dt,
                            angular_velocity[num]);
              }
              break;
            case 3:
#pragma omp parallel for schedule(static)
              for (std::size_t num = 0; num < mNumRigidBody; num++) {
                velocity_k4[num] = mVelocity[num];
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
              for (std::size_t num = 0; num < mNumRigidBody; num++) {
                velocity_k5[num] = mVelocity[num];
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
              for (std::size_t num = 0; num < mNumRigidBody; num++) {
                velocity_k6[num] = mVelocity[num];
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
              for (std::size_t num = 0; num < mNumRigidBody; num++) {
                velocity_k7[num] = mVelocity[num];
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
          for (std::size_t num = 0; num < mNumRigidBody; num++) {
            Vec3 velocity_err =
                (velocity_k1[num] * dc1 + velocity_k3[num] * dc3 +
                 velocity_k4[num] * dc4 + velocity_k5[num] * dc5 +
                 velocity_k6[num] * dc6 + velocity_k7[num] * dc7) *
                dt;

            err += velocity_err.SquareMag();
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
      for (std::size_t num = 0; num < mNumRigidBody; num++) {
        velocity_k1[num] = velocity_k7[num];
        angular_velocity_k1[num] = angular_velocity_k7[num];
      }

      output_file << t << '\t';
      for (std::size_t num = 0; num < mNumRigidBody; num++) {
        for (int i = 0; i < 3; i++)
          output_file << mPosition[num][i] + position_offset[num][i] << '\t';
        float roll, pitch, yaw;
        mOrientation[num].to_euler_angles(roll, pitch, yaw);
        output_file << roll << '\t' << pitch << '\t' << yaw << '\t';
      }
      output_file << std::endl;
    }

    // final output
    output_file << t << '\t';
    for (std::size_t num = 0; num < mNumRigidBody; num++) {
      for (int i = 0; i < 3; i++)
        output_file << mPosition[num][i] + position_offset[num][i] << '\t';
      float roll, pitch, yaw;
      mOrientation[num].to_euler_angles(roll, pitch, yaw);
      output_file << roll << '\t' << pitch << '\t' << yaw << '\t';
    }
    output_file << std::endl;
    output_file.close();
  }
};

#endif