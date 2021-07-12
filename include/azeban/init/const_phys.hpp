#ifndef CONST_PHYS_HPP
#define CONST_PHYS_HPP

#include "initializer.hpp"
#include <azeban/random/random_variable.hpp>
#include <array>
#include <type_traits>


namespace azeban {

template<int Dim>
class ConstPhys final : public Initializer<Dim> {
  using super = Initializer<Dim>;

public:
  template<typename... RVs, typename = std::enable_if_t<(... && std::is_convertible_v<RVs, RandomVariable<real_t>>)>>
  ConstPhys(const RVs&... rvs) : rvs_{rvs...} { }
  ConstPhys(const ConstPhys &) = default;
  ConstPhys(ConstPhys &&) = default;

  virtual ~ConstPhys() = default;

  ConstPhys &operator=(const ConstPhys &) = default;
  ConstPhys &operator=(ConstPhys &&) = default;

protected:
  virtual void do_initialize(const zisa::array_view<real_t, Dim + 1> &u) override {
    const auto init = [&](const zisa::array_view<real_t, Dim + 1> &u_) {
      for (int d = 0 ; d < Dim ; ++d) {
	zisa::shape_t<Dim> shape_view;
	for (int i = 0 ; i < Dim ; ++i) {
	  shape_view[i] = u_.shape(i + 1);
	}
	zisa::array_view<real_t, Dim> view(shape_view, u_.raw() + d * zisa::product(shape_view), u_.memory_location());
	const real_t v = rvs_[d].get();
	zisa::fill(view.raw(), view.memory_location(), view.size(), v);
      }
    };
    if (u.memory_location() == zisa::device_type::cpu) {
      init(u);
    }
    else if (u.memory_location() == zisa::device_type::cuda) {
      zisa::array<real_t, Dim + 1> h_u(u.shape());
      init(h_u);
      zisa::copy(u, h_u);
    }
    else {
      LOG_ERR("Unsupported memory loaction");
    }
  }

  virtual void do_initialize(const zisa::array_view<complex_t, Dim + 1> &u_hat) override {
    const auto init = [&](const zisa::array_view<complex_t, Dim + 1> &u_) {
      const zisa::int_t N = u_.shape(1);
      for (int d = 0 ; d < Dim ; ++d) {
	zisa::shape_t<Dim> shape_view;
	for (int i = 0 ; i < Dim ; ++i) {
	  shape_view[i] = u_.shape(i + 1);
	}
	zisa::array_view<complex_t, Dim> view(shape_view, u_.raw() + d * zisa::product(shape_view), u_.memory_location());
	const real_t v = rvs_[d].get();
	zisa::fill(reinterpret_cast<real_t*>(view.raw()), view.memory_location(), 2 * view.size(), real_t(0));
	view[0] = v * zisa::pow<Dim>(N);
      }
    };
    if (u_hat.memory_location() == zisa::device_type::cpu) {
      init(u_hat);
    }
    else if (u_hat.memory_location() == zisa::device_type::cuda) {
      zisa::array<complex_t, Dim + 1> h_u_hat(u_hat.shape());
      init(h_u_hat);
      zisa::copy(u_hat, h_u_hat);
    }
    else {
      LOG_ERR("Unsupported memory loaction");
    }
  }

private:
  std::array<RandomVariable<real_t>, Dim> rvs_;
};

}


#endif
