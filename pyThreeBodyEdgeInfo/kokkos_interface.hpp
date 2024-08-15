#ifndef KOKKOS_INTERFACE_HPP
#define KOKKOS_INTERFACE_HPP

#include <Kokkos_Core.hpp>

class kokkos_interface {
public:
  kokkos_interface() {
    int argv = 0;
    char arg[] = "neighbor_lists";
    char **argc = (char **)&arg;
    Kokkos::initialize(argv, argc);
    bool success = Kokkos::is_initialized();
    assert(success == true);
  }

  ~kokkos_interface() {}
};

#endif