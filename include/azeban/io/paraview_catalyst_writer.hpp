#ifndef AZEBAN_IO_PARAVIEW_CATALYST_WRITER_HPP_
#define AZEBAN_IO_PARAVIEW_CATALYST_WRITER_HPP_

#if AZEBAN_HAS_CATALYST

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
                         const std::vector<std::vector<std::string>> &scripts,
                         zisa::int_t sample_idx_start = 0);
  ParaviewCatalystWriter(const ParaviewCatalystWriter &) = delete;
  ParaviewCatalystWriter(ParaviewCatalystWriter &&) = default;
  ParaviewCatalystWriter &operator=(const ParaviewCatalystWriter &) = delete;
  ParaviewCatalystWriter &operator=(ParaviewCatalystWriter &&) = default;

  virtual ~ParaviewCatalystWriter() override;

  using super::next_timestep;
  using super::reset;
  virtual void write(const zisa::array_const_view<real_t, Dim + 1> &u,
                     double t) override;
  virtual void write(const zisa::array_const_view<complex_t, Dim + 1> &u_hat,
                     double t) override;
#if AZEBAN_HAS_MPI
  virtual void write(const zisa::array_const_view<real_t, Dim + 1> &u,
                     double t,
                     const Communicator *comm) override;
  virtual void write(const zisa::array_const_view<complex_t, Dim + 1> &u_hat,
                     doublen t,
                     const Communicator *comm) override;
#endif

protected:
  using super::grid_;
  using super::sample_idx_;
  using super::snapshot_idx_;
  using super::snapshot_times_;
};

}

#endif

#endif
