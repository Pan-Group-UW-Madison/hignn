#ifndef _QUATERNION_HPP_
#define _QUATERNION_HPP_

#include <iostream>
#include <vec3.hpp>

#ifdef _WIN32
#define M_PI 3.14159265359
#endif

class quaternion {
private:
  float m_data[4];

public:
  // default constructor
  quaternion() {
    m_data[0] = 0.0;
    m_data[1] = 0.0;
    m_data[2] = 0.0;
    m_data[3] = 0.0;
  }

  quaternion(const quaternion &q) {
    m_data[0] = q.m_data[0];
    m_data[1] = q.m_data[1];
    m_data[2] = q.m_data[2];
    m_data[3] = q.m_data[3];
  }

  // constructor from a vector, means the scalar part would be zero
  quaternion(vec3 vec) {
    m_data[0] = 0.0;
    m_data[1] = vec[0];
    m_data[2] = vec[1];
    m_data[3] = vec[2];
  }

  // constructor from a rotation axis and rotation angle theta
  quaternion(vec3 omega, const float theta) {
    float norm = omega.mag();
    omega = omega * (1.0 / norm);
    float h = theta * norm;
    m_data[0] = cos(h / 2.0);
    m_data[1] = sin(h / 2.0) * omega[0];
    m_data[2] = sin(h / 2.0) * omega[1];
    m_data[3] = sin(h / 2.0) * omega[2];
  }

  // constructor from euler angle, theta1, 2, 3, sequence - x, y, z
  quaternion(float roll, float pitch, float yaw) {
    float cy = cos(yaw * 0.5);
    float sy = sin(yaw * 0.5);
    float cp = cos(pitch * 0.5);
    float sp = sin(pitch * 0.5);
    float cr = cos(roll * 0.5);
    float sr = sin(roll * 0.5);

    m_data[0] = cr * cp * cy + sr * sp * sy;
    m_data[1] = sr * cp * cy - cr * sp * sy;
    m_data[2] = cr * sp * cy + sr * cp * sy;
    m_data[3] = cr * cp * sy - sr * sp * cy;
  }

  quaternion product(quaternion &q) {
    quaternion res;

    res.m_data[0] = m_data[0] * q.m_data[0] - m_data[1] * q.m_data[1] -
                    m_data[2] * q.m_data[2] - m_data[3] * q.m_data[3];
    res.m_data[1] = m_data[0] * q.m_data[1] + m_data[1] * q.m_data[0] +
                    m_data[2] * q.m_data[3] - m_data[3] * q.m_data[2];
    res.m_data[2] = m_data[0] * q.m_data[2] - m_data[1] * q.m_data[3] +
                    m_data[2] * q.m_data[0] + m_data[3] * q.m_data[1];
    res.m_data[3] = m_data[0] * q.m_data[3] + m_data[1] * q.m_data[2] -
                    m_data[2] * q.m_data[1] + m_data[3] * q.m_data[0];

    return res;
  }

  const float q0() const { return m_data[0]; }
  const float q1() const { return m_data[1]; }
  const float q2() const { return m_data[2]; }
  const float q3() const { return m_data[3]; }

  void set_q0(const float q0) { m_data[0] = q0; }
  void set_q1(const float q1) { m_data[1] = q1; }
  void set_q2(const float q2) { m_data[2] = q2; }
  void set_q3(const float q3) { m_data[3] = q3; }

  void Cross(const quaternion &qa, const quaternion &qb) {
    float w = qa.m_data[0] * qb.m_data[0] - qa.m_data[1] * qb.m_data[1] -
              qa.m_data[2] * qb.m_data[2] - qa.m_data[3] * qb.m_data[3];
    float x = qa.m_data[0] * qb.m_data[1] + qa.m_data[1] * qb.m_data[0] -
              qa.m_data[3] * qb.m_data[2] + qa.m_data[2] * qb.m_data[3];
    float y = qa.m_data[0] * qb.m_data[2] + qa.m_data[2] * qb.m_data[0] +
              qa.m_data[3] * qb.m_data[1] - qa.m_data[1] * qb.m_data[3];
    float z = qa.m_data[0] * qb.m_data[3] + qa.m_data[3] * qb.m_data[0] -
              qa.m_data[2] * qb.m_data[1] + qa.m_data[1] * qb.m_data[2];
    m_data[0] = w;
    m_data[1] = x;
    m_data[2] = y;
    m_data[3] = z;
  }

  void to_euler_angles(float &roll, float &pitch, float &yaw) const {
    float sinr_cosp = 2.0 * (m_data[0] * m_data[1] + m_data[2] * m_data[3]);
    float cosr_cosp =
        1.0 - 2.0 * (m_data[1] * m_data[1] + m_data[2] * m_data[2]);
    roll = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    float sinp = 2.0 * (m_data[0] * m_data[2] - m_data[3] * m_data[1]);
    if (std::abs(sinp) >= 1.0)
      pitch = std::copysign(M_PI / 2.0, sinp); // use 90 degrees if out of range
    else
      pitch = std::asin(sinp);

    // yaw (z-axis rotation)
    float siny_cosp = 2.0 * (m_data[0] * m_data[3] + m_data[1] * m_data[2]);
    float cosy_cosp =
        1.0 - 2.0 * (m_data[2] * m_data[2] + m_data[3] * m_data[3]);
    yaw = std::atan2(siny_cosp, cosy_cosp);
  }

  quaternion &operator=(const quaternion &q) {
    m_data[0] = q.m_data[0];
    m_data[1] = q.m_data[1];
    m_data[2] = q.m_data[2];
    m_data[3] = q.m_data[3];

    return *this;
  }

  void conjugate() {
    quaternion q = *this;
    for (int i = 1; i < 4; i++) {
      q.m_data[i] = -q.m_data[i];
    }
  }

  vec3 rotate(const vec3 &vec) {
    float e0e0 = m_data[0] * m_data[0];
    float e1e1 = m_data[1] * m_data[1];
    float e2e2 = m_data[2] * m_data[2];
    float e3e3 = m_data[3] * m_data[3];
    float e0e1 = m_data[0] * m_data[1];
    float e0e2 = m_data[0] * m_data[2];
    float e0e3 = m_data[0] * m_data[3];
    float e1e2 = m_data[1] * m_data[2];
    float e1e3 = m_data[1] * m_data[3];
    float e2e3 = m_data[2] * m_data[3];
    return vec3(
        ((e0e0 + e1e1) * 2.0 - 1.0) * vec[0] + ((e1e2 - e0e3) * 2.0) * vec[1] +
            ((e1e3 + e0e2) * 2.0) * vec[2],
        ((e1e2 + e0e3) * 2.0) * vec[0] + ((e0e0 + e2e2) * 2.0 - 1.0) * vec[1] +
            ((e2e3 - e0e1) * 2) * vec[2],
        ((e1e3 - e0e2) * 2.0) * vec[0] + ((e2e3 + e0e1) * 2.0) * vec[1] +
            ((e0e0 + e3e3) * 2.0 - 1.0) * vec[2]);
  }

  vec3 rotate_back(const vec3 &vec) {
    float e0e0 = +m_data[0] * m_data[0];
    float e1e1 = +m_data[1] * m_data[1];
    float e2e2 = +m_data[2] * m_data[2];
    float e3e3 = +m_data[3] * m_data[3];
    float e0e1 = -m_data[0] * m_data[1];
    float e0e2 = -m_data[0] * m_data[2];
    float e0e3 = -m_data[0] * m_data[3];
    float e1e2 = +m_data[1] * m_data[2];
    float e1e3 = +m_data[1] * m_data[3];
    float e2e3 = +m_data[2] * m_data[3];
    return vec3(
        ((e0e0 + e1e1) * 2.0 - 1.0) * vec[0] + ((e1e2 - e0e3) * 2.0) * vec[1] +
            ((e1e3 + e0e2) * 2.0) * vec[2],
        ((e1e2 + e0e3) * 2.0) * vec[0] + ((e0e0 + e2e2) * 2.0 - 1.0) * vec[1] +
            ((e2e3 - e0e1) * 2.0) * vec[2],
        ((e1e3 - e0e2) * 2.0) * vec[0] + ((e2e3 + e0e1) * 2.0) * vec[1] +
            ((e0e0 + e3e3) * 2.0 - 1.0) * vec[2]);
  }

  void rotate_by_Wabs(vec3 &omega, quaternion &q) {
    float theta = omega.mag();
    quaternion q_delta(omega * (1.0 / theta), theta);
    this->Cross(q, q_delta);
  }
};

vec3 Bracket(const vec3 &v1, const vec3 &v2) { return Cross(v1, v2) * 2.0; }

vec3 dexpinv(const vec3 &u, const vec3 &k) {
  vec3 res;
  vec3 bracket_res = Bracket(u, k);

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