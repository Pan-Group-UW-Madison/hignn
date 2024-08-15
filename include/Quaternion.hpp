#ifndef _Quaternion_Hpp_
#define _Quaternion_Hpp_

#include <iostream>

#include "Vec3.hpp"

#ifdef _WIN32
#define M_PI 3.14159265359
#endif

class Quaternion {
private:
  float mData[4];

public:
  // default constructor
  Quaternion() {
    mData[0] = 0.0;
    mData[1] = 0.0;
    mData[2] = 0.0;
    mData[3] = 0.0;
  }

  Quaternion(const Quaternion &q) {
    mData[0] = q.mData[0];
    mData[1] = q.mData[1];
    mData[2] = q.mData[2];
    mData[3] = q.mData[3];
  }

  // constructor from a vector, means the scalar part would be zero
  Quaternion(Vec3 vec) {
    mData[0] = 0.0;
    mData[1] = vec[0];
    mData[2] = vec[1];
    mData[3] = vec[2];
  }

  // constructor from a rotation axis and rotation angle theta
  Quaternion(Vec3 omega, const float theta) {
    float norm = omega.Mag();
    omega = omega * (1.0 / norm);
    float h = theta * norm;
    mData[0] = cos(h / 2.0);
    mData[1] = sin(h / 2.0) * omega[0];
    mData[2] = sin(h / 2.0) * omega[1];
    mData[3] = sin(h / 2.0) * omega[2];
  }

  // constructor from euler angle, theta1, 2, 3, sequence - x, y, z
  Quaternion(float roll, float pitch, float yaw) {
    float cy = cos(yaw * 0.5);
    float sy = sin(yaw * 0.5);
    float cp = cos(pitch * 0.5);
    float sp = sin(pitch * 0.5);
    float cr = cos(roll * 0.5);
    float sr = sin(roll * 0.5);

    mData[0] = cr * cp * cy + sr * sp * sy;
    mData[1] = sr * cp * cy - cr * sp * sy;
    mData[2] = cr * sp * cy + sr * cp * sy;
    mData[3] = cr * cp * sy - sr * sp * cy;
  }

  Quaternion product(Quaternion &q) {
    Quaternion res;

    res.mData[0] = mData[0] * q.mData[0] - mData[1] * q.mData[1] -
                   mData[2] * q.mData[2] - mData[3] * q.mData[3];
    res.mData[1] = mData[0] * q.mData[1] + mData[1] * q.mData[0] +
                   mData[2] * q.mData[3] - mData[3] * q.mData[2];
    res.mData[2] = mData[0] * q.mData[2] - mData[1] * q.mData[3] +
                   mData[2] * q.mData[0] + mData[3] * q.mData[1];
    res.mData[3] = mData[0] * q.mData[3] + mData[1] * q.mData[2] -
                   mData[2] * q.mData[1] + mData[3] * q.mData[0];

    return res;
  }

  float q0() const {
    return mData[0];
  }
  float q1() const {
    return mData[1];
  }
  float q2() const {
    return mData[2];
  }
  float q3() const {
    return mData[3];
  }

  void set_q0(const float q0) {
    mData[0] = q0;
  }
  void set_q1(const float q1) {
    mData[1] = q1;
  }
  void set_q2(const float q2) {
    mData[2] = q2;
  }
  void set_q3(const float q3) {
    mData[3] = q3;
  }

  void Cross(const Quaternion &qa, const Quaternion &qb) {
    float w = qa.mData[0] * qb.mData[0] - qa.mData[1] * qb.mData[1] -
              qa.mData[2] * qb.mData[2] - qa.mData[3] * qb.mData[3];
    float x = qa.mData[0] * qb.mData[1] + qa.mData[1] * qb.mData[0] -
              qa.mData[3] * qb.mData[2] + qa.mData[2] * qb.mData[3];
    float y = qa.mData[0] * qb.mData[2] + qa.mData[2] * qb.mData[0] +
              qa.mData[3] * qb.mData[1] - qa.mData[1] * qb.mData[3];
    float z = qa.mData[0] * qb.mData[3] + qa.mData[3] * qb.mData[0] -
              qa.mData[2] * qb.mData[1] + qa.mData[1] * qb.mData[2];
    mData[0] = w;
    mData[1] = x;
    mData[2] = y;
    mData[3] = z;
  }

  void to_euler_angles(float &roll, float &pitch, float &yaw) const {
    float sinr_cosp = 2.0 * (mData[0] * mData[1] + mData[2] * mData[3]);
    float cosr_cosp = 1.0 - 2.0 * (mData[1] * mData[1] + mData[2] * mData[2]);
    roll = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    float sinp = 2.0 * (mData[0] * mData[2] - mData[3] * mData[1]);
    if (std::abs(sinp) >= 1.0)
      pitch =
          std::copysign(M_PI / 2.0, sinp);  // use 90 degrees if out of range
    else
      pitch = std::asin(sinp);

    // yaw (z-axis rotation)
    float siny_cosp = 2.0 * (mData[0] * mData[3] + mData[1] * mData[2]);
    float cosy_cosp = 1.0 - 2.0 * (mData[2] * mData[2] + mData[3] * mData[3]);
    yaw = std::atan2(siny_cosp, cosy_cosp);
  }

  Quaternion &operator=(const Quaternion &q) {
    mData[0] = q.mData[0];
    mData[1] = q.mData[1];
    mData[2] = q.mData[2];
    mData[3] = q.mData[3];

    return *this;
  }

  void conjugate() {
    Quaternion q = *this;
    for (int i = 1; i < 4; i++) {
      q.mData[i] = -q.mData[i];
    }
  }

  Vec3 rotate(const Vec3 &vec) {
    float e0e0 = mData[0] * mData[0];
    float e1e1 = mData[1] * mData[1];
    float e2e2 = mData[2] * mData[2];
    float e3e3 = mData[3] * mData[3];
    float e0e1 = mData[0] * mData[1];
    float e0e2 = mData[0] * mData[2];
    float e0e3 = mData[0] * mData[3];
    float e1e2 = mData[1] * mData[2];
    float e1e3 = mData[1] * mData[3];
    float e2e3 = mData[2] * mData[3];
    return Vec3(
        ((e0e0 + e1e1) * 2.0 - 1.0) * vec[0] + ((e1e2 - e0e3) * 2.0) * vec[1] +
            ((e1e3 + e0e2) * 2.0) * vec[2],
        ((e1e2 + e0e3) * 2.0) * vec[0] + ((e0e0 + e2e2) * 2.0 - 1.0) * vec[1] +
            ((e2e3 - e0e1) * 2) * vec[2],
        ((e1e3 - e0e2) * 2.0) * vec[0] + ((e2e3 + e0e1) * 2.0) * vec[1] +
            ((e0e0 + e3e3) * 2.0 - 1.0) * vec[2]);
  }

  Vec3 rotate_back(const Vec3 &vec) {
    float e0e0 = +mData[0] * mData[0];
    float e1e1 = +mData[1] * mData[1];
    float e2e2 = +mData[2] * mData[2];
    float e3e3 = +mData[3] * mData[3];
    float e0e1 = -mData[0] * mData[1];
    float e0e2 = -mData[0] * mData[2];
    float e0e3 = -mData[0] * mData[3];
    float e1e2 = +mData[1] * mData[2];
    float e1e3 = +mData[1] * mData[3];
    float e2e3 = +mData[2] * mData[3];
    return Vec3(
        ((e0e0 + e1e1) * 2.0 - 1.0) * vec[0] + ((e1e2 - e0e3) * 2.0) * vec[1] +
            ((e1e3 + e0e2) * 2.0) * vec[2],
        ((e1e2 + e0e3) * 2.0) * vec[0] + ((e0e0 + e2e2) * 2.0 - 1.0) * vec[1] +
            ((e2e3 - e0e1) * 2.0) * vec[2],
        ((e1e3 - e0e2) * 2.0) * vec[0] + ((e2e3 + e0e1) * 2.0) * vec[1] +
            ((e0e0 + e3e3) * 2.0 - 1.0) * vec[2]);
  }

  void rotate_by_Wabs(Vec3 &omega, Quaternion &q) {
    float theta = omega.Mag();
    Quaternion q_delta(omega * (1.0 / theta), theta);
    this->Cross(q, q_delta);
  }
};

Vec3 Bracket(const Vec3 &v1, const Vec3 &v2) {
  return Cross(v1, v2) * 2.0;
}

Vec3 dexpinv(const Vec3 &u, const Vec3 &k) {
  Vec3 res;
  Vec3 bracket_res = Bracket(u, k);

  res = k - bracket_res * 0.5;
  bracket_res = Bracket(u, bracket_res);
  res = res + bracket_res * static_cast<float>(1.0 / 12.0);
  bracket_res = Bracket(u, Bracket(u, bracket_res));
  res = res - bracket_res * static_cast<float>(1.0 / 720.0);
  bracket_res = Bracket(u, Bracket(u, bracket_res));
  res = res + bracket_res * static_cast<float>(1.0 / 30240.0);
  bracket_res = Bracket(u, Bracket(u, bracket_res));
  res = res - bracket_res * static_cast<float>(1.0 / 1209600.0);

  return res;
}

#endif