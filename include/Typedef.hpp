#ifndef _Typedef_Hpp_
#define _Typedef_Hpp_

#include <Kokkos_Core.hpp>

typedef double Double;
typedef float Float;

typedef Kokkos::View<Float **, Kokkos::DefaultHostExecutionSpace>
    HostFloatMatrix;
typedef Kokkos::View<Float **, Kokkos::DefaultExecutionSpace> DeviceFloatMatrix;

typedef Kokkos::View<Double **, Kokkos::DefaultHostExecutionSpace>
    HostDoubleMatrix;
typedef Kokkos::View<Double **, Kokkos::DefaultExecutionSpace>
    DeviceDoubleMatrix;

#endif