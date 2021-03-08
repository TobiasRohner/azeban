#include <azeban/equations/burgers.hpp>
#include <azeban/equations/incompressible_euler.hpp>
#include <azeban/evolution/evolution.hpp>
#include <azeban/fft.hpp>
#include <azeban/grid.hpp>
#include <azeban/simulation.hpp>
#include <cstdlib>
#include <zisa/config.hpp>
#include <zisa/cuda/memory/cuda_array.hpp>
#include <zisa/io/hdf5_serial_writer.hpp>
#include <zisa/memory/array.hpp>

using namespace azeban;

int main() {
  static constexpr int dim_v = 2;
  static constexpr int n_vars = 2;
  static constexpr azeban::real_t t_final = 5;
  static constexpr int n_frames = 600;

  azeban::Grid<dim_v> grid(256);
  const zisa::int_t N_phys = grid.N_phys;
  const zisa::int_t N_fourier = grid.N_fourier;

  zisa::HDF5SerialWriter hdf5_writer("result.hdf5");

  auto u_host = grid.make_array_phys(n_vars, zisa::device_type::cpu);
  auto u_device = grid.make_array_phys(n_vars, zisa::device_type::cuda);
  auto u_hat_device = grid.make_array_fourier(n_vars, zisa::device_type::cuda);

  auto fft = make_fft<dim_v>(u_hat_device, u_device);

  /*
  for (zisa::int_t i = 0; i < N_phys; ++i) {
    u_host[i] = zisa::sin(2 * zisa::pi * i / N_phys);
  }
  zisa::copy(u_device, u_host);
  fft->forward();

  auto equation = std::make_shared<Burgers<SmoothCutoff1D>>(
      grid, SmoothCutoff1D(0.0 / N_phys, 1), zisa::device_type::cuda);
  */

  // Periodic shear layer
  const azeban::real_t rho = zisa::pi / 15;
  const azeban::real_t delta = 0.05;
  for (zisa::int_t i = 0; i < N_phys; ++i) {
    for (zisa::int_t j = 0; j < N_phys; ++j) {
      const azeban::real_t x
          = 2 * zisa::pi * static_cast<azeban::real_t>(i) / N_phys;
      const azeban::real_t y
          = 2 * zisa::pi * static_cast<azeban::real_t>(j) / N_phys;
      if (y <= zisa::pi) {
        u_host(0, i, j) = std::tanh((y - zisa::pi / 2) / rho);
      } else {
        u_host(0, i, j) = std::tanh((3 * zisa::pi / 2 - y) / rho);
      }
      u_host(1, i, j) = delta * zisa::sin(x);
    }
  }
  zisa::copy(u_device, u_host);
  fft->forward();

  auto equation = std::make_shared<IncompressibleEuler<2, SmoothCutoff1D>>(
      grid, SmoothCutoff1D(0.05 / N_phys, 1), zisa::device_type::cuda);

  CFL cfl(grid, 0.1);
  auto timestepper = std::make_shared<SSP_RK3<complex_t, dim_v>>(
      zisa::device_type::cuda, grid.shape_fourier(n_vars), equation);
  auto simulation
      = Simulation<complex_t, dim_v>(u_hat_device, cfl, timestepper);

  zisa::save(hdf5_writer, u_host, std::to_string(0));
  for (int i = 0; i < n_frames; ++i) {
    std::cerr << i << std::endl;
    simulation.simulate_for(t_final / n_frames);

    zisa::copy(u_hat_device, simulation.u());
    fft->backward();
    zisa::copy(u_host, u_device);
    for (zisa::int_t i = 0; i < zisa::product(u_host.shape()); ++i) {
      u_host[i] /= zisa::product(u_host.shape()) / u_host.shape(0);
    }
    zisa::save(hdf5_writer, u_host, std::to_string(i + 1));
  }

  return EXIT_SUCCESS;
}
