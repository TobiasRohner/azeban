#ifndef INIT_FROM_FILE_HPP_
#define INIT_FROM_FILE_HPP_

#include "initializer.hpp"
#include <vector>

namespace azeban {

template <int Dim>
class InitFromFile : public Initializer<Dim> {
  using super = Initializer<Dim>;

public:
  InitFromFile(const std::string &experiment, const std::string &time, zisa::int_t sample_idx_start=0)
      : sample_(sample_idx_start), experiment_(experiment), time_(time) {}
  InitFromFile(const InitFromFile &) = default;
  InitFromFile(InitFromFile &&) = default;

  virtual ~InitFromFile() override = default;

  InitFromFile &operator=(const InitFromFile &) = default;
  InitFromFile &operator=(InitFromFile &&) = default;

  virtual void initialize(const zisa::array_view<real_t, Dim + 1> &u) override;
  virtual void
  initialize(const zisa::array_view<complex_t, Dim + 1> &u_hat) override;

protected:
  virtual void
  do_initialize(const zisa::array_view<real_t, Dim + 1> & /*u*/) override {}
  virtual void do_initialize(
      const zisa::array_view<complex_t, Dim + 1> & /*u_hat*/) override {}

private:
  zisa::int_t sample_;
  std::string experiment_;
  std::string time_;

  std::string filename() const;
  std::vector<size_t> get_dims(int ncid, int varid) const;
  void read_component(int ncid,
                      const std::string &name,
                      const zisa::array_view<real_t, Dim> &u) const;
};

}

#endif
