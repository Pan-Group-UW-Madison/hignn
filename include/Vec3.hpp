#ifndef _Vec3_Hpp_
#define _Vec3_Hpp_

#include <cmath>
#include <fstream>
#include <vector>

template <class T>
class Triple {
  T data[3];

public:
  Triple() {
    data[0] = 0;
    data[1] = 0;
    data[2] = 0;
  }

  Triple(T first, T second, T third) {
    data[0] = first;
    data[1] = second;
    data[2] = third;
  }

  Triple(const Triple<T> &t) {
    data[0] = t.data[0];
    data[1] = t.data[1];
    data[2] = t.data[2];
  }

  T &operator[](const int i) {
    return data[i];
  }

  const T operator[](const int i) const {
    return data[i];
  }

  void operator+=(const Triple<T> &y) {
    data[0] += y[0];
    data[1] += y[1];
    data[2] += y[2];
  }

  void operator-=(const Triple<T> &y) {
    data[0] -= y[0];
    data[1] -= y[1];
    data[2] -= y[2];
  }

  void operator=(const Triple<T> &y) {
    data[0] = y[0];
    data[1] = y[1];
    data[2] = y[2];
  }

  void operator*=(const float a) {
    data[0] *= a;
    data[1] *= a;
    data[2] *= a;
  }

  bool operator>(const Triple<T> &y) {
    return ((data[0] > y[0]) || (data[1] > y[1]) || (data[2] > y[2]));
  }

  bool operator<(const Triple<T> &y) {
    return ((data[0] < y[0]) || (data[1] < y[1]) || (data[2] < y[2]));
  }

  const Triple<T> operator-(const Triple<T> &y) const {
    return Triple<T>(data[0] - y[0], data[1] - y[1], data[2] - y[2]);
  }

  Triple<T> operator*(float a) {
    return Triple<T>(a * data[0], a * data[1], a * data[2]);
  }

  float Mag() {
    return sqrt(data[0] * data[0] + data[1] * data[1] + data[2] * data[2]);
  }

  float SquareMag() {
    return (data[0] * data[0] + data[1] * data[1] + data[2] * data[2]);
  }

  float Cdot(const Triple<T> &y) {
    return y[0] * data[0] + y[1] * data[1] + y[2] * data[2];
  }
};

template <class T>
Triple<T> operator+(const Triple<T> &xScalar, const Triple<T> &y) {
  return Triple<T>(xScalar[0] + y[0], xScalar[1] + y[1], xScalar[2] + y[2]);
}

template <class T>
float MaxMag(const std::vector<Triple<T>> &maxOf) {
  float maxM = 0.0;
  for (int i = 0; i < maxOf.size(); i++) {
    maxM = MAX(maxM, maxOf[i].mag());
  }

  return maxM;
}

typedef Triple<float> Vec3;

Vec3 Cross(const Vec3 &v1, const Vec3 &v2) {
  Vec3 res(v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2],
           v1[0] * v2[1] - v1[1] * v2[0]);
  return res;
}

#endif