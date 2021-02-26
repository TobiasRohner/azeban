#include <cstdlib>
#include <azeban/fft.hpp>
#include <azeban/simulation.hpp>
#include <azeban/equations/burgers.hpp>
#include <azeban/evolution/forward_euler.hpp>
#include <zisa/config.hpp>
#include <zisa/memory/array.hpp>
#include <zisa/cuda/memory/cuda_array.hpp>



using namespace azeban;


int main() {
  zisa::int_t N_phys = 128;
  zisa::int_t N_fourier = N_phys/2 + 1;

  auto u_host = zisa::array<real_t, 1>(zisa::shape_t<1>{N_phys});
  auto u_device = zisa::cuda_array<real_t, 1>(zisa::shape_t<1>{N_phys});
  auto u_hat_device = zisa::cuda_array<complex_t, 1>(zisa::shape_t<1>{N_fourier});

  auto fft = azeban::make_fft<1>(zisa::array_view<complex_t, 1>(u_hat_device),
			         zisa::array_view<real_t, 1>(u_device));

  for (zisa::int_t i = 0 ; i < N_phys ; ++i) {
    u_host[i] = i < N_phys/4 ? 1 : 0;
  }
  zisa::copy(u_device, u_host);
  fft->forward();

  auto timestepper = std::make_shared<azeban::ForwardEuler<azeban::complex_t, 1>>();
  auto equation = std::make_shared<azeban::Burgers<Step1D>>(N_phys, azeban::Step1D(1, 0), zisa::device_type::cuda);
  auto simulation = azeban::Simulation<complex_t, 1>(zisa::array_const_view<complex_t, 1>(u_hat_device),
						     equation,
						     timestepper);

  simulation.simulate_until(0.001, 0.0001);

  // Ugly, but normal copy doesn't work for some reason
  zisa::internal::copy(u_hat_device.raw(), u_hat_device.device(),
		       simulation.u().raw(), simulation.u().memory_location(),
		       N_fourier);
  fft->backward();
  zisa::copy(u_host, u_device);
  zisa::internal::copy(u_host.raw(), u_host.device(),
		       u_device.raw(), u_device.device(),
		       N_phys);

  for (zisa::int_t i = 0 ; i < N_phys ; ++i) {
    std::cout << u_host[i]/N_phys << "\n";
  }

  return EXIT_SUCCESS;
}
