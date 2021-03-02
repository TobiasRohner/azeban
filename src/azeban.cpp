#include <azeban/equations/burgers.hpp>
#include <azeban/evolution/evolution.hpp>
#include <azeban/fft.hpp>
#include <azeban/simulation.hpp>
#include <cstdlib>
#include <zisa/config.hpp>
#include <zisa/cuda/memory/cuda_array.hpp>
#include <zisa/memory/array.hpp>

using namespace azeban;

int main() {
  zisa::int_t N_phys = 1024 * 2;
  zisa::int_t N_fourier = N_phys / 2 + 1;

  auto u_host = zisa::array<real_t, 1>(zisa::shape_t<1>{N_phys});
  auto u_device = zisa::cuda_array<real_t, 1>(zisa::shape_t<1>{N_phys});
  auto u_hat_device
      = zisa::cuda_array<complex_t, 1>(zisa::shape_t<1>{N_fourier});

  auto fft = make_fft<1>(zisa::array_view<complex_t, 1>(u_hat_device),
                         zisa::array_view<real_t, 1>(u_device));

  for (zisa::int_t i = 0; i < N_phys; ++i) {
    u_host[i] = zisa::sin(2 * zisa::pi / N_phys * i);
  }
  zisa::copy(u_device, u_host);
  fft->forward();

  CFL cfl(0.5);
  auto equation = std::make_shared<Burgers<SmoothCutoff1D>>(
      N_phys, SmoothCutoff1D(0. / N_phys, 1), zisa::device_type::cuda);
  auto timestepper = std::make_shared<SSP_RK2<complex_t, 1>>(
      zisa::device_type::cuda, zisa::shape_t<1>(N_fourier), equation);
  auto simulation = Simulation<complex_t, 1>(
      zisa::array_const_view<complex_t, 1>(u_hat_device), cfl, timestepper);

  for (int i = 0; i < 1000; ++i) {
    std::cerr << i << std::endl;
    simulation.simulate_for(0.1 / 1000);

    // Ugly, but normal copy doesn't work for some reason
    /*
    zisa::internal::copy(u_hat_device.raw(), u_hat_device.device(),
                         simulation.u().raw(), simulation.u().memory_location(),
                         N_fourier);
    */
    zisa::copy(u_hat_device,
               zisa::array_const_view<complex_t, 1>(simulation.u()));
    fft->backward();
    zisa::copy(u_host, u_device);
    zisa::internal::copy(u_host.raw(),
                         u_host.device(),
                         u_device.raw(),
                         u_device.device(),
                         N_phys);

    for (zisa::int_t i = 0; i < N_phys; ++i) {
      std::cout << u_host[i] / N_phys << "\n";
    }
    std::cout << "\n\n";
  }

  return EXIT_SUCCESS;
}
