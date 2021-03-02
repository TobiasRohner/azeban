#ifndef TIME_INTEGRATOR_H_
#define TIME_INTEGRATOR_H_

#include <azeban/config.hpp>
#include <azeban/equations/equation.hpp>
#include <zisa/config.hpp>
#include <zisa/memory/array.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

template <typename Scalar, int Dim>
class TimeIntegrator {
public:
  using scalar_t = Scalar;
  static constexpr int dim_v = Dim;

  TimeIntegrator() = delete;
  TimeIntegrator(zisa::device_type device,
                 const std::shared_ptr<Equation<scalar_t, dim_v>> &equation)
      : device_(device), equation_(equation) {}
  TimeIntegrator(const TimeIntegrator &) = default;
  TimeIntegrator(TimeIntegrator &&) = default;

  virtual ~TimeIntegrator() = default;

  TimeIntegrator &operator=(const TimeIntegrator &) = default;
  TimeIntegrator &operator=(TimeIntegrator &&) = default;

  virtual void integrate(real_t dt, const zisa::array_view<scalar_t, dim_v> &u)
      = 0;

  zisa::device_type memory_location() const { return device_; }
  const auto &equation() const { return equation_; }

protected:
  std::shared_ptr<Equation<scalar_t, dim_v>> equation_;
  zisa::device_type device_;
};

}

#endif
