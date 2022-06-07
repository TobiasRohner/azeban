#ifndef AZEBAN_IO_PARAVIEW_CATALYST_WRITER_HPP_
#define AZEBAN_IO_PARAVIEW_CATALYST_WRITER_HPP_

#include <azeban/grid.hpp>
#include <azeban/io/writer.hpp>
#include <string>
#include <vector>

namespace azeban {

template <int Dim>
class ParaviewCatalystWriter : public Writer<Dim> {
  using super = Writer<Dim>;

public:
  ParaviewCatalystWriter(const Grid<Dim> &grid,
                         const std::vector<real_t> &snapshot_times_,
                         const std::vector<std::string> &scripts,
                         zisa::int_t sample_idx_start = 0);
  ParaviewCatalystWriter(const ParaviewCatalystWriter &) = delete;
  ParaviewCatalystWriter(ParaviewCatalystWriter &&) = default;
  ParaviewCatalystWriter &operator=(const ParaviewCatalystWriter &) = delete;
  ParaviewCatalystWriter &operator=(ParaviewCatalystWriter &&) = default;

  virtual ~ParaviewCatalystWriter() override;

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
  Grid<Dim> grid_;
  std::vector<real_t> snapshot_times_;
  zisa::int_t sample_idx_;
  zisa::int_t snapshot_idx_;
};

}

#endif
