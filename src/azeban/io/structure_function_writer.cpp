#include <azeban/io/structure_function_writer.hpp>
#include <azeban/operations/structure_function.hpp>
#include <azeban/operations/structure_function_functionals.hpp>
#include <azeban/profiler.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <zisa/math/basic_functions.hpp>

namespace azeban {

template <int Dim, typename Function>
StructureFunctionWriter<Dim, Function>::StructureFunctionWriter(
    const std::string &path,
    const Grid<Dim> &grid,
    const std::vector<real_t> &snapshot_times,
    zisa::int_t sample_idx_start,
    const std::string &name,
    const Function &func,
    ssize_t max_h)
    : super(grid, snapshot_times, sample_idx_start),
      path_(path),
      name_(name),
      func_(func),
      max_h_(max_h) {}

template <int Dim, typename Function>
void StructureFunctionWriter<Dim, Function>::write(
    const zisa::array_const_view<real_t, Dim + 1> &u, real_t t) {
  ProfileHost pofile("StructureFunctionWriter::write");
  ZISA_UNUSED(t);
  const std::vector<real_t> S = structure_function<Dim>(u, max_h_, func_);
  std::ofstream file(path_ + "/" + name_ + "_sample_"
                     + std::to_string(sample_idx_) + "_time_"
                     + std::to_string(snapshot_idx_) + ".txt");
  for (real_t E : S) {
    file << std::setprecision(std::numeric_limits<real_t>::max_digits10) << E
         << '\t';
  }
  ++snapshot_idx_;
}

template <int Dim, typename Function>
void StructureFunctionWriter<Dim, Function>::write(
    const zisa::array_const_view<complex_t, Dim + 1> &u_hat, real_t t) {
  ZISA_UNUSED(u_hat);
  ZISA_UNUSED(t);
}

#if AZEBAN_HAS_MPI
template <int Dim, typename Function>
void StructureFunctionWriter<Dim, Function>::write(
    const zisa::array_const_view<real_t, Dim + 1> &u,
    real_t t,
    const Communicator *comm) {
  ZISA_UNUSED(u);
  ZISA_UNUSED(t);
  ZISA_UNUSED(comm);
}

template <int Dim, typename Function>
void StructureFunctionWriter<Dim, Function>::write(
    const zisa::array_const_view<complex_t, Dim + 1> &u_hat,
    real_t t,
    const Communicator *comm) {
  ZISA_UNUSED(u_hat);
  ZISA_UNUSED(t);
  ZISA_UNUSED(comm);
}
#endif

template class StructureFunctionWriter<1, SFCubeFunctional>;
template class StructureFunctionWriter<2, SFCubeFunctional>;
template class StructureFunctionWriter<3, SFCubeFunctional>;
template class StructureFunctionWriter<1, SFThirdOrderFunctional>;
template class StructureFunctionWriter<2, SFThirdOrderFunctional>;
template class StructureFunctionWriter<3, SFThirdOrderFunctional>;
template class StructureFunctionWriter<1, SFLongitudinalFunctional>;
template class StructureFunctionWriter<2, SFLongitudinalFunctional>;
template class StructureFunctionWriter<3, SFLongitudinalFunctional>;

}
