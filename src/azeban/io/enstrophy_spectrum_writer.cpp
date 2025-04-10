#include <azeban/io/enstrophy_spectrum_writer.hpp>
#include <azeban/operations/enstrophy_spectrum.hpp>
#include <azeban/profiler.hpp>
#include <experimental/filesystem>
#include <iomanip>
#include <limits>
#if AZEBAN_HAS_MPI
#include <azeban/mpi/mpi_types.hpp>
#endif

namespace azeban {

template <int Dim>
EnstrophySpectrumWriter<Dim>::EnstrophySpectrumWriter(
    const std::string &path,
    const Grid<Dim> &grid,
    const std::vector<double> &snapshot_times,
    int sample_idx_start)
    : super(grid, snapshot_times, sample_idx_start), path_(path) {
#if AZEBAN_HAS_MPI
  samples_comm_ = MPI_COMM_WORLD;
#endif
  if (!std::experimental::filesystem::exists(path)) {
    std::experimental::filesystem::create_directories(path);
  }
}

#if AZEBAN_HAS_MPI
template <int Dim>
EnstrophySpectrumWriter<Dim>::EnstrophySpectrumWriter(
    const std::string &path,
    const Grid<Dim> &grid,
    const std::vector<double> &snapshot_times,
    int sample_idx_start,
    const Communicator *comm)
    : super(grid, snapshot_times, sample_idx_start), path_(path) {
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int local_rank = comm->rank();
  int color = local_rank == 0 ? 0 : MPI_UNDEFINED;
  MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &samples_comm_);
  if (local_rank == 0) {
    if (!std::experimental::filesystem::exists(path)) {
      std::experimental::filesystem::create_directories(path);
    }
  }
}
#endif

template <int Dim>
void EnstrophySpectrumWriter<Dim>::reset() {
  file_.close();
  super::reset();
}

template <int Dim>
void EnstrophySpectrumWriter<Dim>::write(
    const zisa::array_const_view<real_t, Dim + 1> &u, double t) {
  ZISA_UNUSED(u);
  ZISA_UNUSED(t);
}

template <int Dim>
void EnstrophySpectrumWriter<Dim>::write(
    const zisa::array_const_view<complex_t, Dim + 1> &u_hat, double t) {
  ProfileHost pofile("EnstrophySpectrumWriter::write");
  ZISA_UNUSED(t);
  const std::vector<real_t> spectrum = enstrophy_spectrum(grid_, u_hat);
  if (!file_.is_open()) {
    file_.open(path_ + "/enstrophy_" + std::to_string(sample_idx_) + ".txt");
  }
  for (real_t E : spectrum) {
    file_ << std::setprecision(std::numeric_limits<real_t>::max_digits10) << E
          << '\t';
  }
  file_ << '\n';
  ++snapshot_idx_;
}

#if AZEBAN_HAS_MPI
template <int Dim>
void EnstrophySpectrumWriter<Dim>::write(
    const zisa::array_const_view<real_t, Dim + 1> &u,
    double t,
    const Communicator *comm) {
  ZISA_UNUSED(u);
  ZISA_UNUSED(t);
  ZISA_UNUSED(comm);
}

template <int Dim>
void EnstrophySpectrumWriter<Dim>::write(
    const zisa::array_const_view<complex_t, Dim + 1> &u_hat,
    double t,
    const Communicator *comm) {
  ProfileHost pofile("EnstrophySpectrumWriter::write");
  ZISA_UNUSED(t);
  const std::vector<real_t> spectrum
      = enstrophy_spectrum(grid_, u_hat, comm->get_mpi_comm());
  if (comm->rank() == 0) {
    if (!file_.is_open()) {
      file_.open(path_ + "/enstrophy_" + std::to_string(sample_idx_) + ".txt");
    }
    for (real_t E : spectrum) {
      file_ << std::setprecision(std::numeric_limits<real_t>::max_digits10) << E
            << '\t';
    }
    file_ << '\n';
  }
  ++snapshot_idx_;
}
#endif

template class EnstrophySpectrumWriter<1>;
template class EnstrophySpectrumWriter<2>;
template class EnstrophySpectrumWriter<3>;

}
