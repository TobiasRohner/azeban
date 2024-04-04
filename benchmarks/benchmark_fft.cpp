#include <azeban/grid.hpp>
#include <azeban/operations/fft_mpi_factory.hpp>
#include <chrono>
#include <iostream>
#include <vector>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  const int N = std::stoi(argv[1]);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int M = 128;

  azeban::Grid<3> grid(N);
  zisa::array<azeban::complex_t, 4> u_hat_host
      = grid.make_array_fourier(1, zisa::device_type::cpu, MPI_COMM_WORLD);
  zisa::array<azeban::complex_t, 4> u_hat_device
      = grid.make_array_fourier(1, zisa::device_type::cuda, MPI_COMM_WORLD);
  zisa::array<azeban::real_t, 4> u_host
      = grid.make_array_phys(1, zisa::device_type::cpu, MPI_COMM_WORLD);
  zisa::array<azeban::real_t, 4> u_device
      = grid.make_array_phys(1, zisa::device_type::cuda, MPI_COMM_WORLD);

  azeban::Communicator comm(MPI_COMM_WORLD);
  auto fft = azeban::make_fft_mpi<3>(u_hat_device, u_device, &comm);

  std::vector<double> times_forward;
  std::vector<double> times_backward;
  for (int i = 0; i < M; ++i) {
    zisa::fill(u_host, azeban::real_t{0});
    if (rank == 0) {
      u_host(0, 0, 0, 0) = 1;
    }
    zisa::copy(u_device, u_host);

    {
      const auto start = std::chrono::steady_clock::now();
      fft->forward();
      const auto end = std::chrono::steady_clock::now();
      const double elapsed
          = std::chrono::duration_cast<std::chrono::duration<double>>(end
                                                                      - start)
                .count();
      times_forward.push_back(elapsed);
    }

    {
      const auto start = std::chrono::steady_clock::now();
      fft->backward();
      const auto end = std::chrono::steady_clock::now();
      const double elapsed
          = std::chrono::duration_cast<std::chrono::duration<double>>(end
                                                                      - start)
                .count();
      times_backward.push_back(elapsed);
    }
  }

  double max_forward = times_forward[0];
  double min_forward = times_forward[0];
  double mean_forward = 0;
  for (double dt : times_forward) {
    if (dt > max_forward) {
      max_forward = dt;
    }
    if (dt < min_forward) {
      min_forward = dt;
    }
    mean_forward += dt;
  }
  mean_forward /= M;

  double max_backward = times_backward[0];
  double min_backward = times_backward[0];
  double mean_backward = 0;
  for (double dt : times_backward) {
    if (dt > max_backward) {
      max_backward = dt;
    }
    if (dt < min_backward) {
      min_backward = dt;
    }
    mean_backward += dt;
  }
  mean_backward /= M;

  if (rank == 0) {
    std::cout << "Forward:  min=" << min_forward << "s, max=" << max_forward
              << "s, mean=" << mean_forward << "s\n";
    std::cout << "Backward: min=" << min_backward << "s, max=" << max_backward
              << "s, mean=" << mean_backward << "s\n";
  }

  MPI_Finalize();
  return 0;
}
