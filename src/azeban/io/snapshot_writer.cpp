#include <azeban/io/snapshot_writer.hpp>
#include <azeban/operations/fft_factory.hpp>
#include <azeban/profiler.hpp>
#include <vector>
#include <zisa/memory/array.hpp>
#if AZEBAN_HAS_MPI
#include <azeban/operations/fft_mpi_factory.hpp>
#endif

namespace azeban {

template <int Dim>
void SnapshotWriter<Dim>::write_snapshot(const Simulation<Dim> &simulation,
                                         zisa::int_t sample_idx) {
  AZEBAN_PROFILE_START("SnapshotWriter::write_snapshot");
  const Grid<Dim> &grid = simulation.grid();
  auto u_host
      = grid.make_array_phys(simulation.n_vars(), zisa::device_type::cpu);
  auto u_hat_host
      = grid.make_array_fourier(simulation.n_vars(), zisa::device_type::cpu);
  auto fft = make_fft<Dim>(u_hat_host, u_host, FFT_BACKWARD);
  zisa::copy(u_hat_host, simulation.u());
  fft->backward();
  for (zisa::int_t i = 0; i < zisa::product(u_host.shape()); ++i) {
    u_host[i] /= zisa::product(u_host.shape()) / u_host.shape(0);
  }
  do_write_snapshot(sample_idx, simulation.time(), u_host);
  AZEBAN_PROFILE_STOP("SnapshotWriter::write_snapshot");
}

#if AZEBAN_HAS_MPI
template <int Dim>
void SnapshotWriter<Dim>::write_snapshot(const Simulation<Dim> &simulation,
                                         zisa::int_t sample_idx,
                                         MPI_Comm comm) {
  AZEBAN_PROFILE_START("SnapshotWriter::write_snapshot", comm);
  if constexpr (Dim > 1) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const Grid<Dim> &grid = simulation.grid();

    std::vector<int> cnts(size);
    std::vector<int> displs(size);
    for (int r = 0; r < size; ++r) {
      cnts[r] = zisa::pow<Dim - 1>(grid.N_phys)
                * (grid.N_phys / size
                   + (zisa::integer_cast<zisa::int_t>(r) < grid.N_phys % size));
    }
    displs[0] = 0;
    for (int r = 1; r < size; ++r) {
      displs[r] = displs[r - 1] + cnts[r - 1];
    }
    const zisa::int_t n_elems_per_component_glob
        = zisa::product(grid.shape_phys(1));
    const zisa::int_t n_elems_per_component_loc
        = zisa::product(grid.shape_phys(1, comm));

    zisa::array<real_t, Dim + 1> u_init;
    if (rank == 0) {
      u_init
          = grid.make_array_phys(simulation.n_vars(), zisa::device_type::cpu);
    }
    auto u_host = grid.make_array_phys(
        simulation.n_vars(), zisa::device_type::cpu, comm);
    auto u_device = grid.make_array_phys(
        simulation.n_vars(), zisa::device_type::cuda, comm);
    auto u_hat_device = grid.make_array_fourier(
        simulation.n_vars(), zisa::device_type::cuda, comm);
    auto fft = make_fft_mpi<Dim>(u_hat_device,
                                 u_device,
                                 comm,
                                 FFT_BACKWARD,
                                 simulation.equation()->get_fft_work_area());
    zisa::copy(u_hat_device, simulation.u());
    fft->backward();
    zisa::copy(u_host, u_device);
    for (zisa::int_t i = 0; i < zisa::product(u_host.shape()); ++i) {
      u_host[i] /= zisa::pow<Dim>(grid.N_phys);
    }
    std::vector<MPI_Request> reqs(simulation.n_vars());
    for (zisa::int_t i = 0; i < simulation.n_vars(); ++i) {
      MPI_Igatherv(u_host.raw() + i * n_elems_per_component_loc,
                   cnts[rank],
                   mpi_type<real_t>(),
                   u_init.raw() + i * n_elems_per_component_glob,
                   cnts.data(),
                   displs.data(),
                   mpi_type<real_t>(),
                   0,
                   comm,
                   &reqs[i]);
    }
    MPI_Waitall(simulation.n_vars(), reqs.data(), MPI_STATUSES_IGNORE);

    if (rank == 0) {
      do_write_snapshot(sample_idx, simulation.time(), u_init);
    }
  }
  AZEBAN_PROFILE_STOP("SnapshotWriter::write_snapshot", comm);
}
#endif

template class SnapshotWriter<1>;
template class SnapshotWriter<2>;
template class SnapshotWriter<3>;

}
