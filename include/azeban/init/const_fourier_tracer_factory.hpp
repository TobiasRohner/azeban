#ifndef CONST_FOURIER_TRACER_FACTORY_HPP_
#define CONST_FOURIER_TRACER_FACTORY_HPP_

#include "const_fourier_tracer.hpp"
#include <fmt/core.h>
#include <nlohmann/json.hpp>


namespace azeban {

template<int Dim>
std::shared_ptr<Initializer<Dim>>
make_const_fourier_tracer(const nlohmann::json &config) {
  if (!config.contains("rho")) {
    fmt::print(stderr, "ConstFourier initialization is missing parameter \"rho\"");
    exit(1);
  }
  real_t rho = config["rho"];
  return std::make_shared<ConstFourierTracer<Dim>>(rho);
}

}


#endif
