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
  zisa::int_t N_phys = 1024;
  zisa::int_t N_fourier = N_phys / 2 + 1;

  zisa::HDF5SerialWriter hdf5_writer("result.hdf5");

  auto u_host = zisa::array<real_t, 1>(zisa::shape_t<1>{N_phys});
  auto u_device = zisa::cuda_array<real_t, 1>(zisa::shape_t<1>{N_phys});
  auto u_hat_device
      = zisa::cuda_array<complex_t, 1>(zisa::shape_t<1>{N_fourier});

  auto fft = make_fft<1>(zisa::array_view<complex_t, 1>(u_hat_device),
                         zisa::array_view<real_t, 1>(u_device));

  for (zisa::int_t i = 0; i < N_phys; ++i) {
    u_host[i] = zisa::sin(2 * zisa::pi / N_phys * i);
    //u_host[i] = i < N_phys / 4 ? 1 : 0;
  }
  zisa::copy(u_device, u_host);
  fft->forward();

  /*
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
        u_host(0, i, j) = u0 * std::tanh(80 * (y - 0.25));
      } else {
        u_host(0, i, j) = u0 * std::tanh(80 * (0.75 - y));
      }
      u_host(1, i, j) = delta * u0 * zisa::sin(2 * zisa::pi * (x + 0.25));
    }
  }
  zisa::copy(u_device, u_host);
  fft->forward();
  */

  CFL cfl(0.5);
  auto equation = std::make_shared<Burgers<SmoothCutoff1D>>(
      N_phys, SmoothCutoff1D(0. / N_phys, 0.1), zisa::device_type::cuda);
  auto timestepper = std::make_shared<SSP_RK2<complex_t, 1>>(
      zisa::device_type::cuda, zisa::shape_t<1>(N_fourier), equation);
  auto simulation = Simulation<complex_t, 1>(
      zisa::array_const_view<complex_t, 1>(u_hat_device), cfl, timestepper);

  /*
  CFL cfl(0.5);
  auto equation = std::make_shared<IncompressibleEuler<2, SmoothCutoff1D>>(
      N_phys, SmoothCutoff1D(0.5 / N_phys, 1), zisa::device_type::cuda);
  auto timestepper = std::make_shared<SSP_RK2<complex_t, 3>>(
      zisa::device_type::cuda,
      zisa::shape_t<3>(2, N_phys, N_fourier),
      equation);
  auto simulation = Simulation<complex_t, 3>(
      zisa::array_const_view<complex_t, 3>(u_hat_device), cfl, timestepper);
  */

  zisa::save(hdf5_writer, u_host, std::to_string(real_t(0)));
  for (int i = 0; i < 1000; ++i) {
    std::cerr << i << std::endl;
    simulation.simulate_for(0.25 / 1000);

    zisa::copy(u_hat_device, simulation.u());
    fft->backward();
    zisa::copy(u_host, u_device);
    for (zisa::int_t i = 0; i < N_phys; ++i) {
      u_host[i] /= zisa::product(u_host.shape());// / u_host.shape(0);
    }
    zisa::save(hdf5_writer, u_host, std::to_string(simulation.time()));

    for (zisa::int_t i = 0; i < N_phys; ++i) {
      std::cout << u_host[i] << "\n";
    }
    std::cout << "\n\n";
  }

  return EXIT_SUCCESS;
}
