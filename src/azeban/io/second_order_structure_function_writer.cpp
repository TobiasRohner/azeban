#include <azeban/io/second_order_structure_function_writer.hpp>
#include <azeban/operations/second_order_structure_function.hpp>
#include <azeban/profiler.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#if AZEBAN_HAS_MPI
#include <azeban/mpi/mpi_types.hpp>
#endif

namespace azeban {

template <int Dim, typename SF>
SecondOrderStructureFunctionWriter<Dim, SF>::SecondOrderStructureFunctionWriter(
    const std::string &path,
    const Grid<Dim> &grid,
    const std::vector<real_t> &snapshot_times,
    zisa::int_t sample_idx_start)
    : super(grid, snapshot_times, sample_idx_start), path_(path) {
#if AZEBAN_HAS_MPI
  samples_comm_ = MPI_COMM_WORLD;
#endif
}

#if AZEBAN_HAS_MPI
template <int Dim, typename SF>
SecondOrderStructureFunctionWriter<Dim, SF>::SecondOrderStructureFunctionWriter(
    const std::string &path,
    const Grid<Dim> &grid,
    const std::vector<real_t> &snapshot_times,
    zisa::int_t sample_idx_start,
    const Communicator *comm)
    : super(grid, snapshot_times, sample_idx_start), path_(path) {
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int local_rank = comm->rank();
  int color = local_rank == 0 ? 0 : MPI_UNDEFINED;
  MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &samples_comm_);
}
#endif

template <int Dim, typename SF>
void SecondOrderStructureFunctionWriter<Dim, SF>::write(
    const zisa::array_const_view<real_t, Dim + 1> &u, real_t t) {
  ZISA_UNUSED(u);
  ZISA_UNUSED(t);
}

template <int Dim, typename SF>
void SecondOrderStructureFunctionWriter<Dim, SF>::write(
    const zisa::array_const_view<complex_t, Dim + 1> &u_hat, real_t t) {
  ProfileHost pofile("SecondOrderStructureFunctionWriter::write");
  ZISA_UNUSED(t);
  const std::vector<real_t> S = SF::eval(grid_, u_hat);
  std::ofstream file(path_ + "/S2_" + std::to_string(sample_idx_) + "_time_"
                     + std::to_string(snapshot_idx_) + ".txt");
  for (real_t E : S) {
    file << std::setprecision(std::numeric_limits<real_t>::max_digits10) << E
         << '\t';
  }
  ++snapshot_idx_;
}

#if AZEBAN_HAS_MPI
template <int Dim, typename SF>
void SecondOrderStructureFunctionWriter<Dim, SF>::write(
    const zisa::array_const_view<real_t, Dim + 1> &u,
    real_t t,
    const Communicator *comm) {
  ZISA_UNUSED(u);
  ZISA_UNUSED(t);
  ZISA_UNUSED(comm);
}

template <int Dim, typename SF>
void SecondOrderStructureFunctionWriter<Dim, SF>::write(
    const zisa::array_const_view<complex_t, Dim + 1> &u_hat,
    real_t t,
    const Communicator *comm) {
  ProfileHost pofile("SecondOrderStructureFunctionWriter::write");
  ZISA_UNUSED(t);
  const std::vector<real_t> S = SF::eval(grid_, u_hat, comm->get_mpi_comm());
  if (comm->rank() == 0) {
    std::ofstream file(path_ + "/S2_" + std::to_string(sample_idx_) + "_time_"
                       + std::to_string(snapshot_idx_) + ".txt");
    for (real_t E : S) {
      file << std::setprecision(std::numeric_limits<real_t>::max_digits10) << E
           << '\t';
    }
  }
  ++snapshot_idx_;
}
#endif

template class SecondOrderStructureFunctionWriter<1, detail::SF2Exact>;
template class SecondOrderStructureFunctionWriter<2, detail::SF2Exact>;
template class SecondOrderStructureFunctionWriter<3, detail::SF2Exact>;
template class SecondOrderStructureFunctionWriter<1, detail::SF2Approx>;
template class SecondOrderStructureFunctionWriter<2, detail::SF2Approx>;
template class SecondOrderStructureFunctionWriter<3, detail::SF2Approx>;

}
