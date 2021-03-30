#ifndef INIT_3D_FROM_2D_H_
#define INIT_3D_FROM_2D_H_

#include "initializer.hpp"

namespace azeban {

class Init3DFrom2D final : public Initializer<3> {
  using super = Initializer<3>;

public:
  Init3DFrom2D(zisa::int_t dim, const std::shared_ptr<Initializer<2>> &init2d)
      : dim_(dim), init2d_(init2d) {}
  Init3DFrom2D(const Init3DFrom2D &) = default;
  Init3DFrom2D(Init3DFrom2D &&) = default;

  virtual ~Init3DFrom2D() override = default;

  Init3DFrom2D &operator=(const Init3DFrom2D &) = default;
  Init3DFrom2D &operator=(Init3DFrom2D &&) = default;

protected:
  virtual void do_initialize(const zisa::array_view<real_t, 4> &u) override;
  virtual void
  do_initialize(const zisa::array_view<complex_t, 4> &u_hat) override;

private:
  zisa::int_t dim_;
  std::shared_ptr<Initializer<2>> init2d_;
};

}

#endif
