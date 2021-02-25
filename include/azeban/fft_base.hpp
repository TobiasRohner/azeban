#ifndef FFT_BASE_H_WIQBB
#define FFT_BASE_H_WIQBB

#include <azeban/config.hpp>
#include <zisa/memory/array.hpp>

namespace azeban {


template<int Dim>
class FFT {
public:
  static constexpr int dim_v = Dim;

  // This constructor assumes that we have scalar valued data elements
  FFT(const zisa::array_view<complex_t, dim_v> &u_hat, const zisa::array_view<real_t, dim_v> &u)
    : FFT(increase_array_rank(u_hat), increase_array_rank(u)) { }

  // This constructor assumes that we have vector valued data elements
  FFT(const zisa::array_view<complex_t, dim_v+1> &u_hat, const zisa::array_view<real_t, dim_v+1> &u)
      : u_hat_(u_hat),
	u_(u),
	data_dim_(u_.shape()[0]) {
    assert(u_hat.shape()[0] == u.shape()[0] && "Dimensionality of data elements must be equal!");
  }

  FFT() = default;
  FFT(const FFT&) = default;
  FFT(FFT&&) = default;

  virtual ~FFT() = default;

  FFT& operator=(const FFT&) = default;
  FFT& operator=(FFT&&) = default;

  virtual void forward() = 0;
  virtual void backward() = 0;

  decltype(auto) shape() const { return u_.shape(); }
  decltype(auto) shape(zisa::int_t i) const { return u_.shape(i); }

  const zisa::array_view<complex_t, dim_v+1> &u_hat() { return u_hat_; }
  const zisa::array_const_view<complex_t, dim_v+1> u_hat() const { return u_hat_; }
  const zisa::array_view<real_t, dim_v+1> &u() { return u_; }
  const zisa::array_const_view<real_t, dim_v+1> u() const { return u_; }

protected:
  zisa::array_view<complex_t, dim_v+1> u_hat_;
  zisa::array_view<real_t, dim_v+1> u_;
  zisa::int_t data_dim_;

  template<typename T, int D>
  static zisa::array_view<T, D+1> increase_array_view_rank(const zisa::array_view<T, D> &arr) {
    zisa::shape_t<D+1> shape_new;
    shape_new[0] = 1;
    for (int i = 0 ; i < D ; ++i) {
      shape_new[i+1] = arr.shape()[i];
    }
    return zisa::array_view<T, D+1>(shape_new, arr.raw(), arr.memory_location());
  }
};


}

#endif
