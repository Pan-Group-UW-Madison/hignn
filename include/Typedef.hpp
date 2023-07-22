#ifndef _Typedef_Hpp_
#define _Typedef_Hpp_

#include <Kokkos_Core.hpp>

typedef double Double;
typedef float Float;

typedef Kokkos::View<Float **, Kokkos::DefaultHostExecutionSpace>
    HostFloatMatrix;
typedef Kokkos::View<Float **, Kokkos::DefaultExecutionSpace> DeviceFloatMatrix;

typedef Kokkos::View<Float *, Kokkos::DefaultHostExecutionSpace>
    HostFloatVector;
typedef Kokkos::View<Float *, Kokkos::DefaultExecutionSpace> DeviceFloatVector;

typedef Kokkos::View<Double **, Kokkos::DefaultHostExecutionSpace>
    HostDoubleMatrix;
typedef Kokkos::View<Double **, Kokkos::DefaultExecutionSpace>
    DeviceDoubleMatrix;

typedef Kokkos::View<Double *, Kokkos::DefaultHostExecutionSpace>
    HostDoubleVector;
typedef Kokkos::View<Double *, Kokkos::DefaultExecutionSpace>
    DeviceDoubleVector;

typedef Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> HostIntVector;
typedef Kokkos::View<int *, Kokkos::DefaultExecutionSpace> DeviceIntVector;

typedef Kokkos::View<int **, Kokkos::DefaultHostExecutionSpace> HostIntMatrix;
typedef Kokkos::View<int **, Kokkos::DefaultExecutionSpace> DeviceIntMatrix;

typedef Kokkos::View<std::size_t **, Kokkos::DefaultHostExecutionSpace>
    HostIndexMatrix;
typedef Kokkos::View<std::size_t **, Kokkos::DefaultExecutionSpace>
    DeviceIndexMatrix;

typedef Kokkos::View<std::size_t *, Kokkos::DefaultHostExecutionSpace>
    HostIndexVector;
typedef Kokkos::View<std::size_t *, Kokkos::DefaultExecutionSpace>
    DeviceIndexVector;

#endif