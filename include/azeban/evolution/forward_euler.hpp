#ifndef FORWARD_EULER_H_
#define FORWARD_EULER_H_

#include "time_integrator.hpp"
#include <azeban/operations/axpy.hpp>



namespace azeban {


template<typename Scalar, int Dim>
class ForwardEuler final : public TimeIntegrator<Scalar, Dim> {
  using super = TimeIntegrator<Scalar, Dim>;

public:
  using scalar_t = Scalar;
  static constexpr int dim_v = Dim;

  ForwardEuler() = default;
  ForwardEuler(const ForwardEuler&) = default;
  ForwardEuler(ForwardEuler&&) = default;

  virtual ~ForwardEuler() override = default;

  ForwardEuler& operator=(const ForwardEuler&) = default;
  ForwardEuler& operator=(ForwardEuler&&) = default;

protected:

  virtual void integrate_impl(real_t dt,
			      const zisa::array_view<scalar_t, dim_v> &u,
			      const zisa::array_const_view<scalar_t, dim_v> &du) override {
    axpy(scalar_t(dt), du, u);
  }
};


}



#endif
