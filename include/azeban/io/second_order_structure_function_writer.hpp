#ifndef AZEBAN_IO_SECOND_ORDER_STRUCTURE_FUNCTION_WRITER_HPP_
#define AZEBAN_IO_SECOND_ORDER_STRUCTURE_FUNCTION_WRITER_HPP_

#include <azeban/io/writer.hpp>
#include <azeban/operations/second_order_structure_function.hpp>
#include <cmath>

namespace azeban {

namespace detail {

struct SF2Exact {
  template <int Dim>
  static std::vector<real_t>
  eval(const Grid<Dim> &grid,
       const zisa::array_const_view<complex_t, Dim + 1> &u_hat) {
    return second_order_structure_function_exact<Dim>(grid, u_hat);
  }
#if AZEBAN_HAS_MPI
  template <int Dim>
  static std::vector<real_t>
  eval(const Grid<Dim> &grid,
       const zisa::array_const_view<complex_t, Dim + 1> &u_hat,
       MPI_Comm comm) {
    return second_order_structure_function_exact<Dim>(grid, u_hat, comm);
  }
#endif
};

struct SF2Approx {
  template <int Dim>
  static std::vector<real_t>
  eval(const Grid<Dim> &grid,
       const zisa::array_const_view<complex_t, Dim + 1> &u_hat) {
    return second_order_structure_function_approx<Dim>(grid, u_hat);
  }
#if AZEBAN_HAS_MPI
  template <int Dim>
  static std::vector<real_t>
  eval(const Grid<Dim> &grid,
       const zisa::array_const_view<complex_t, Dim + 1> &u_hat,
       MPI_Comm comm) {
    return second_order_structure_function_approx<Dim>(grid, u_hat, comm);
  }
#endif
};

}

template <int Dim, typename SF>
class SecondOrderStructureFunctionWriter : public Writer<Dim> {
  using super = Writer<Dim>;

public:
  SecondOrderStructureFunctionWriter(const std::string &path,
                                     const Grid<Dim> &grid,
                                     const std::vector<double> &snapshot_times,
                                     zisa::int_t sample_idx_start);
#if AZEBAN_HAS_MPI
  SecondOrderStructureFunctionWriter(const std::string &path,
                                     const Grid<Dim> &grid,
                                     const std::vector<double> &snapshot_times,
                                     zisa::int_t sample_idx_start,
                                     const Communicator *comm);
#endif
  SecondOrderStructureFunctionWriter(const SecondOrderStructureFunctionWriter &)
      = default;
  SecondOrderStructureFunctionWriter(SecondOrderStructureFunctionWriter &&)
      = default;

  virtual ~SecondOrderStructureFunctionWriter() override = default;

  SecondOrderStructureFunctionWriter &
  operator=(const SecondOrderStructureFunctionWriter &)
      = default;
  SecondOrderStructureFunctionWriter &
  operator=(SecondOrderStructureFunctionWriter &&)
      = default;

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
                     double t,
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
using SecondOrderStructureFunctionWriter2Exact
    = SecondOrderStructureFunctionWriter<Dim, detail::SF2Exact>;

template <int Dim>
using SecondOrderStructureFunctionWriter2Approx
    = SecondOrderStructureFunctionWriter<Dim, detail::SF2Approx>;

}

#endif
