#include <Kokkos_Core.hpp>

typedef Kokkos::View<double **, Kokkos::DefaultExecutionSpace> View2D;

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  Kokkos::DefaultExecutionSpace().print_configuration(std::cout);

  {
    // Training batch size
    const int64_t N = 64;
    // Tensor input dimension
    const int64_t D_in = 1000;
    // Tensor hidden dimension
    const int64_t H = 100;
    // NN output dimension
    const int64_t D_out = 10;
    // Total number of steps
    const int64_t tstep = 1000;

    View2D foo1("Foo1", N, D_in);
  }

  Kokkos::finalize();

  return 0;
}