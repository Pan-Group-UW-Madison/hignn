#ifndef _VEC3_HPP_
#define _VEC3_HPP_

#include <cmath>
#include <fstream>
#include <vector>

template <class T> class triple {
  T data[3];

public:
  triple() {
    data[0] = 0;
    data[1] = 0;
    data[2] = 0;
  }

  triple(T first, T second, T third) {
    data[0] = first;
    data[1] = second;
    data[2] = third;
  }

  triple(const triple<T> &t) {
    data[0] = t.data[0];
    data[1] = t.data[1];
    data[2] = t.data[2];
  }

  T &operator[](const int i) { return data[i]; }

  const T operator[](const int i) const { return data[i]; }

  void operator+=(const triple<T> &y) {
    data[0] += y[0];
    data[1] += y[1];
    data[2] += y[2];
  }

  void operator-=(const triple<T> &y) {
    data[0] -= y[0];
    data[1] -= y[1];
    data[2] -= y[2];
  }

  void operator=(const triple<T> &y) {
    data[0] = y[0];
    data[1] = y[1];
    data[2] = y[2];
  }

  void operator*=(const float a) {
    data[0] *= a;
    data[1] *= a;
    data[2] *= a;
  }

  bool operator>(const triple<T> &y) {
    return ((data[0] > y[0]) || (data[1] > y[1]) || (data[2] > y[2]));
  }

  bool operator<(const triple<T> &y) {
    return ((data[0] < y[0]) || (data[1] < y[1]) || (data[2] < y[2]));
  }

  const triple<T> operator-(const triple<T> &y) const {
    return triple<T>(data[0] - y[0], data[1] - y[1], data[2] - y[2]);
  }

  triple<T> operator*(float a) {
    return triple<T>(a * data[0], a * data[1], a * data[2]);
  }

  float mag() {
    return sqrt(data[0] * data[0] + data[1] * data[1] + data[2] * data[2]);
  }

  float sqrmag() {
    return (data[0] * data[0] + data[1] * data[1] + data[2] * data[2]);
  }

  float cdot(const triple<T> &y) {
    return y[0] * data[0] + y[1] * data[1] + y[2] * data[2];
  }
};

template <class T>
triple<T> operator+(const triple<T> &xScalar, const triple<T> &y) {
  return triple<T>(xScalar[0] + y[0], xScalar[1] + y[1], xScalar[2] + y[2]);
}

template <class T> float maxmag(const std::vector<triple<T>> &maxof) {
  float maxm = 0.0;
  for (int i = 0; i < maxof.size(); i++) {
    maxm = MAX(maxm, maxof[i].mag());
  }
  return maxm;
}

typedef triple<float> vec3;

vec3 Cross(const vec3 &v1, const vec3 &v2) {
  vec3 res(v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2],
           v1[0] * v2[1] - v1[1] * v2[0]);
  return res;
}

#endif