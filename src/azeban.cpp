#include <azeban/grid.hpp>
#include <azeban/equations/burgers.hpp>
#include <azeban/equations/incompressible_euler.hpp>
#include <azeban/evolution/evolution.hpp>
#include <azeban/fft.hpp>
#include <azeban/simulation.hpp>
#include <cstdlib>
#include <zisa/config.hpp>
#include <zisa/cuda/memory/cuda_array.hpp>
#include <zisa/io/hdf5_serial_writer.hpp>
#include <zisa/memory/array.hpp>

using namespace azeban;

int main() {
  azeban::Grid<2> grid(256);
  const zisa::int_t N_phys = grid.N_phys;
  const zisa::int_t N_fourier = grid.N_fourier;

  zisa::HDF5SerialWriter hdf5_writer("result.hdf5");

  auto u_host = zisa::array<real_t, 3>(zisa::shape_t<3>{2, N_phys, N_phys});
  auto u_device
      = zisa::cuda_array<real_t, 3>(zisa::shape_t<3>{2, N_phys, N_phys});
  auto u_hat_device
      = zisa::cuda_array<complex_t, 3>(zisa::shape_t<3>{2, N_phys, N_fourier});

  auto fft = make_fft<2>(zisa::array_view<complex_t, 3>(u_hat_device),
                         zisa::array_view<real_t, 3>(u_device));

  // Periodic shear layer
  const azeban::real_t u0 = 1;
  const azeban::real_t delta = 0.5;
  for (zisa::int_t i = 0; i < N_phys; ++i) {
    for (zisa::int_t j = 0; j < N_phys; ++j) {
      const azeban::real_t x = static_cast<azeban::real_t>(i) / N_phys;
      const azeban::real_t y = static_cast<azeban::real_t>(j) / N_phys;
      if (y <= 0.5) {
        u_host(0, i, j) = u0 * std::tanh(2 * (y - 0.25));
      } else {
        u_host(0, i, j) = u0 * std::tanh(2 * (0.75 - y));
      }
      u_host(1, i, j) = delta * u0 * zisa::sin(2 * zisa::pi * (x + 0.25));
    }
  }
  zisa::copy(u_device, u_host);
  fft->forward();

  CFL cfl(grid, 0.5);
  auto equation = std::make_shared<IncompressibleEuler<2, SmoothCutoff1D>>(
      grid, SmoothCutoff1D(0.5 / N_phys, 1), zisa::device_type::cuda);
  auto timestepper = std::make_shared<SSP_RK2<complex_t, 2>>(
      zisa::device_type::cuda,
      zisa::shape_t<3>(2, N_phys, N_fourier),
      equation);
  auto simulation = Simulation<complex_t, 2>(
      zisa::array_const_view<complex_t, 3>(u_hat_device), cfl, timestepper);

  zisa::save(hdf5_writer, u_host, std::to_string(0));
  for (int i = 0; i < 1000; ++i) {
    std::cerr << i << std::endl;
    simulation.simulate_for(10. / 1000);

    zisa::copy(u_hat_device, simulation.u());
    fft->backward();
    zisa::copy(u_host, u_device);
    for (zisa::int_t i = 0; i < N_phys; ++i) {
      u_host[i] /= zisa::product(u_host.shape()) / u_host.shape(0);
    }
    zisa::save(hdf5_writer, u_host, std::to_string(i + 1));
  }

  return EXIT_SUCCESS;
}
