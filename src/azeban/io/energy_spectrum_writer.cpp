#include <azeban/io/energy_spectrum_writer.hpp>
#include <azeban/operations/energy_spectrum.hpp>
#include <azeban/profiler.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#if AZEBAN_HAS_MPI
#include <azeban/mpi/mpi_types.hpp>
#endif

namespace azeban {

template <int Dim>
EnergySpectrumWriter<Dim>::EnergySpectrumWriter(
    const std::string &path,
    const Grid<Dim> &grid,
    const std::vector<real_t> &snapshot_times,
    int sample_idx_start)
    : super(grid, snapshot_times, sample_idx_start), path_(path) {
#if AZEBAN_HAS_MPI
  samples_comm_ = MPI_COMM_WORLD;
#endif
}

#if AZEBAN_HAS_MPI
template <int Dim>
EnergySpectrumWriter<Dim>::EnergySpectrumWriter(
    const std::string &path,
    const Grid<Dim> &grid,
    const std::vector<real_t> &snapshot_times,
    int sample_idx_start,
    const Communicator *comm)
    : super(grid, snapshot_times, sample_idx_start), path_(path) {
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int local_rank = comm->rank();
  int color = local_rank == 0 ? 0 : MPI_UNDEFINED;
  MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &samples_comm_);
}
#endif

template <int Dim>
void EnergySpectrumWriter<Dim>::write(
    const zisa::array_const_view<real_t, Dim + 1> &u, real_t t) {
  ZISA_UNUSED(u);
  ZISA_UNUSED(t);
}

template <int Dim>
void EnergySpectrumWriter<Dim>::write(
    const zisa::array_const_view<complex_t, Dim + 1> &u_hat, real_t t) {
  ProfileHost pofile("EnergySpectrumWriter::write");
  ZISA_UNUSED(t);
  const std::vector<real_t> spectrum = energy_spectrum(grid_, u_hat);
#if 0 // AZEBAN_HAS_MPI
  std::vector<real_t> spectrum_avg(spectrum.size(), 0);
  int reduction_size, reduction_rank;
  MPI_Comm_size(samples_comm_, &reduction_size);
  MPI_Comm_rank(samples_comm_, &reduction_rank);
  MPI_Reduce(spectrum.data(),
             spectrum_avg.data(),
             spectrum.size(),
             mpi_type<real_t>(),
             MPI_SUM,
             0,
             samples_comm_);
  for (real_t &v : spectrum_avg) {
    v /= reduction_size;
  }
  if (reduction_rank == 0) {
    std::ofstream file(path_ + "/energy_" + std::to_string(sample_idx_)
                       + "_time_" + std::to_string(snapshot_idx_) + ".txt");
    for (real_t E : spectrum_avg) {
      file << std::setw(std::numeric_limits<real_t>::max_digits10) << E << '\t';
    }
  }
#else
  std::ofstream file(path_ + "/energy_" + std::to_string(sample_idx_) + "_time_"
                     + std::to_string(snapshot_idx_) + ".txt");
  for (real_t E : spectrum) {
    file << std::setw(std::numeric_limits<real_t>::max_digits10) << E << '\t';
  }
#endif
  ++snapshot_idx_;
}

#if AZEBAN_HAS_MPI
template <int Dim>
void EnergySpectrumWriter<Dim>::write(
    const zisa::array_const_view<real_t, Dim + 1> &u,
    real_t t,
    const Communicator *comm) {
  ZISA_UNUSED(u);
  ZISA_UNUSED(t);
  ZISA_UNUSED(comm);
}

template <int Dim>
void EnergySpectrumWriter<Dim>::write(
    const zisa::array_const_view<complex_t, Dim + 1> &u_hat,
    real_t t,
    const Communicator *comm) {
  ProfileHost pofile("EnergySpectrumWriter::write");
  ZISA_UNUSED(t);
  const std::vector<real_t> spectrum
      = energy_spectrum(grid_, u_hat, comm->get_mpi_comm());
  /*
  std::vector<real_t> spectrum_avg(spectrum.size(), 0);
  const int local_rank = comm->rank();
  if (local_rank == 0) {
    int reduction_size, reduction_rank;
    MPI_Comm_size(samples_comm_, &reduction_size);
    MPI_Comm_rank(samples_comm_, &reduction_rank);
    MPI_Reduce(spectrum.data(),
               spectrum_avg.data(),
               spectrum.size(),
               mpi_type<real_t>(),
               MPI_SUM,
               0,
               samples_comm_);
    for (real_t &v : spectrum_avg) {
      v /= reduction_size;
    }
    if (reduction_rank == 0) {
      std::ofstream file(path_ + "/energy_" + std::to_string(sample_idx_)
                         + "_time_" + std::to_string(snapshot_idx_) + ".txt");
      for (real_t E : spectrum_avg) {
        file << std::setw(std::numeric_limits<real_t>::max_digits10) << E
             << '\t';
      }
    }
  }
  */
  if (comm->rank() == 0) {
    std::ofstream file(path_ + "/energy_" + std::to_string(sample_idx_) + "_time_"
		       + std::to_string(snapshot_idx_) + ".txt");
    for (real_t E : spectrum) {
      file << std::setw(std::numeric_limits<real_t>::max_digits10) << E << '\t';
    }
  }
  ++snapshot_idx_;
}
#endif

template class EnergySpectrumWriter<1>;
template class EnergySpectrumWriter<2>;
template class EnergySpectrumWriter<3>;

}
