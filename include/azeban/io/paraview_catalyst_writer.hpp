#ifndef AZEBAN_IO_PARAVIEW_CATALYST_WRITER_HPP_
#define AZEBAN_IO_PARAVIEW_CATALYST_WRITER_HPP_

#include <azeban/io/writer.hpp>

namespace azeban {

template <int Dim>
class ParaviewCatalystWriter : public Writer<Dim> {
  using super = Writery<Dim>;

public:
  ParaviewCatalystWriter(const std::string &script);
  ParaviewCatalystWriter(const ParaviewCatalystWriter &) = default;
  ParaviewCatalystWriter &operator=(const ParaviewCatalystWriter &) = default;

  virtual ~ParaviewCatalystWriter() override = default;

  virtual void reset() override;
  virtual real_t next_timestep() const override;
  virtual void write(const zisa::array_const_view<real_t, Dim + 1> &u,
                     real_t t) override;
  virtual void write(const zisa::array_const_view<complex_t, Dim + 1> &u_hat,
                     real_t t) override;
#if AZEBAN_HAS_MPI
  virtual void write(const zisa::array_const_view<real_t, Dim + 1> &u,
                     real_t t,
                     const Communicator *comm) override;
  virtual void write(const zisa::array_const_view<complex_t, Dim + 1> &u_hat,
                     real_t t,
                     const Communicator *comm) override;
#endif

private:
};

}

#endif
