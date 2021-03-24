#ifndef FFT_BASE_H_WIQBB
#define FFT_BASE_H_WIQBB

#include <azeban/config.hpp>
#include <zisa/memory/array.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

enum fft_direction { FFT_FORWARD = 1, FFT_BACKWARD = 2 };

template <int Dim>
class FFT {
public:
  static constexpr int dim_v = Dim;

  FFT(const zisa::array_view<complex_t, dim_v + 1> &u_hat,
      const zisa::array_view<real_t, dim_v + 1> &u,
      int direction)
      : u_hat_(u_hat), u_(u), data_dim_(u_.shape()[0]), direction_(direction) {
    assert(u_hat.shape()[0] == u.shape()[0]
           && "Dimensionality of data elements must be equal!");
  }

  FFT() = default;
  FFT(const FFT &) = default;
  FFT(FFT &&) = default;

  virtual ~FFT() = default;

  FFT &operator=(const FFT &) = default;
  FFT &operator=(FFT &&) = default;

  virtual void forward() = 0;
  virtual void backward() = 0;

  decltype(auto) shape() const { return u_.shape(); }
  decltype(auto) shape(zisa::int_t i) const { return u_.shape(i); }

  const zisa::array_view<complex_t, dim_v + 1> &u_hat() { return u_hat_; }
  const zisa::array_const_view<complex_t, dim_v + 1> u_hat() const {
    return u_hat_;
  }
  const zisa::array_view<real_t, dim_v + 1> &u() { return u_; }
  const zisa::array_const_view<real_t, dim_v + 1> u() const { return u_; }

protected:
  zisa::array_view<complex_t, dim_v + 1> u_hat_;
  zisa::array_view<real_t, dim_v + 1> u_;
  zisa::int_t data_dim_;
  int direction_;
};

}

#endif
