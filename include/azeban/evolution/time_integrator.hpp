#ifndef TIME_INTEGRATOR_H_
#define TIME_INTEGRATOR_H_

#include <zisa/config.hpp>
#include <zisa/memory/array.hpp>
#include <zisa/memory/array_view.hpp>
#include <azeban/config.hpp>


namespace azeban {


template<typename Scalar, int Dim>
class TimeIntegrator {
public:
  using scalar_t = Scalar;
  static constexpr int dim_v = Dim;

  TimeIntegrator() { }
  TimeIntegrator(const TimeIntegrator&) = default;
  TimeIntegrator(TimeIntegrator&&) = default;

  virtual ~TimeIntegrator() = default;

  TimeIntegrator& operator=(const TimeIntegrator&) = default;
  TimeIntegrator& operator=(TimeIntegrator&&) = default;

  virtual void integrate(real_t dt,
			 const zisa::array_view<scalar_t, dim_v> &u,
			 const zisa::array_const_view<scalar_t, dim_v> &du) = 0;
};


}



#endif
