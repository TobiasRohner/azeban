#ifndef AZEBAN_IO_STRUCTURE_FUNCTION_WRITER_HPP_
#define AZEBAN_IO_STRUCTURE_FUNCTION_WRITER_HPP_

#include <azeban/io/writer.hpp>
#include <azeban/operations/structure_function.hpp>
#include <cmath>

namespace azeban {

namespace detail {

struct SFExact {
  template <int Dim>
  static std::vector<real_t>
  eval(const Grid<Dim> &grid,
       const zisa::array_const_view<complex_t, Dim + 1> &u_hat) {
    return structure_function_exact<Dim>(grid, u_hat);
  }
#if AZEBAN_HAS_MPI
  template <int Dim>
  static std::vector<real_t>
  eval(const Grid<Dim> &grid,
       const zisa::array_const_view<complex_t, Dim + 1> &u_hat,
       MPI_Comm comm) {
    return structure_function_exact<Dim>(grid, u_hat, comm);
  }
#endif
};

struct SFApprox {
  template <int Dim>
  static std::vector<real_t>
  eval(const Grid<Dim> &grid,
       const zisa::array_const_view<complex_t, Dim + 1> &u_hat) {
    return structure_function_approx<Dim>(grid, u_hat);
  }
#if AZEBAN_HAS_MPI
  template <int Dim>
  static std::vector<real_t>
  eval(const Grid<Dim> &grid,
       const zisa::array_const_view<complex_t, Dim + 1> &u_hat,
       MPI_Comm comm) {
    return structure_function_approx<Dim>(grid, u_hat, comm);
  }
#endif
};

}

template <int Dim, typename SF>
class StructureFunctionWriter : public Writer<Dim> {
  using super = Writer<Dim>;

public:
  StructureFunctionWriter(const std::string &path,
                          const Grid<Dim> &grid,
                          const std::vector<real_t> &snapshot_times,
                          zisa::int_t sample_idx_start);
#if AZEBAN_HAS_MPI
  StructureFunctionWriter(const std::string &path,
                          const Grid<Dim> &grid,
                          const std::vector<real_t> &snapshot_times,
                          zisa::int_t sample_idx_start,
                          const Communicator *comm);
#endif
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
#if AZEBAN_HAS_MPI
  MPI_Comm samples_comm_;
#endif
};

template <int Dim>
using StructureFunctionWriterExact
    = StructureFunctionWriter<Dim, detail::SFExact>;

template <int Dim>
using StructureFunctionWriterApprox
    = StructureFunctionWriter<Dim, detail::SFApprox>;

}

#endif
