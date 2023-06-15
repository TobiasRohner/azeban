#ifndef AZEBAN_IO_STRUCTURE_FUNCTION_WRITER_HPP_
#define AZEBAN_IO_STRUCTURE_FUNCTION_WRITER_HPP_

#include <azeban/io/writer.hpp>
#include <azeban/operations/structure_function.hpp>
#include <azeban/operations/structure_function_functionals.hpp>
#include <cmath>

namespace azeban {

template <int Dim, typename Function>
class StructureFunctionWriter : public Writer<Dim> {
  using super = Writer<Dim>;

public:
  StructureFunctionWriter(const std::string &path,
                          const Grid<Dim> &grid,
                          const std::vector<real_t> &snapshot_times,
                          zisa::int_t sample_idx_start,
                          const std::string &name,
                          const Function &func,
                          ssize_t max_h);

  StructureFunctionWriter(const StructureFunctionWriter &) = default;
  StructureFunctionWriter(StructureFunctionWriter &&) = default;

  virtual ~StructureFunctionWriter() override = default;

  StructureFunctionWriter &operator=(const StructureFunctionWriter &) = default;
  StructureFunctionWriter &operator=(StructureFunctionWriter &&) = default;

  using super::next_timestep;
  using super::reset;
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

protected:
  using super::grid_;
  using super::sample_idx_;
  using super::snapshot_idx_;
  using super::snapshot_times_;

private:
  std::string path_;
  std::string name_;
  Function func_;
  ssize_t max_h_;
};

}

#endif
