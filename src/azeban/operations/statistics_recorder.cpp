#include <azeban/operations/statistics_recorder.hpp>

namespace azeban {

template <int Dim>
StatisticsRecorder<Dim>::StatisticsRecorder(const Grid<Dim> &grid,
                                            bool has_tracer)
    : grid_(grid),
      has_tracer_(has_tracer),
      count_(0),
      Mk_(grid.shape_phys(Dim + has_tracer), zisa::device_type::cpu),
      Sk_(grid.shape_phys(Dim + has_tracer), zisa::device_type::cpu),
      is_finalized_(false) {
  zisa::fill(Mk_, real_t{0});
  zisa::fill(Sk_, real_t{0});
}

#if AZEBAN_HAS_MPI
template <int Dim>
StatisticsRecorder<Dim>::StatisticsRecorder(const Grid<Dim> &grid,
                                            bool has_tracer,
                                            Communicator *comm)
    : grid_(grid),
      has_tracer_(has_tracer),
      count_(0),
      Mk_(grid.shape_phys(Dim + has_tracer, comm), zisa::device_type::cpu),
      Sk_(grid.shape_phys(Dim + has_tracer, comm), zisa::device_type::cpu),
      is_finalized_(false) {
  zisa::fill(Mk_, real_t{0});
  zisa::fill(Sk_, real_t{0});
}
#endif

template <int Dim>
void StatisticsRecorder<Dim>::update(
    const zisa::array_const_view<real_t, Dim + 1> &u) {
  LOG_ERR_IF(is_finalized_,
             "Cannot update an already finalized statistics recorder");
  count_ += 1;
  for (zisa::int_t i = 0; i < u.size(); ++i) {
    const real_t u_new = u[i];
    real_t mean = Mk_[i];
    const real_t delta1 = u_new - mean;
    mean += delta1 / count_;
    const real_t delta2 = u_new - mean;
    Mk_[i] = mean;
    Sk_[i] += delta1 * delta2;
  }
}

template <int Dim>
void StatisticsRecorder<Dim>::finalize() {
  LOG_ERR_IF(is_finalized_,
             "Cannot finalize an already finalized statistics recorder");
  for (zisa::int_t i = 0; i < Sk_.size(); ++i) {
    Sk_[i] /= count_;
  }
  is_finalized_ = true;
}

#if AZEBAN_HAS_MPI
template <int Dim>
void StatisticsRecorder<Dim>::finalize(MPI_Comm comm) {
  static constexpr size_t BUFFER_SIZE = 4 * 1024 * 1024 / sizeof(real_t);
  LOG_ERR_IF(is_finalized_,
             "Cannot finalize an already finalized statistics recorder");
  int size;
  int rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  std::unique_ptr<zisa::int_t[]> counts = std::make_unique<zisa::int_t[]>(size);
  std::unique_ptr<real_t[]> buffer_Mk = std::make_unique<real_t[]>(BUFFER_SIZE);
  std::unique_ptr<real_t[]> buffer_Sk = std::make_unique<real_t[]>(BUFFER_SIZE);
  MPI_Allgather(&count_,
                1,
                mpi_type<zisa::int_t>(),
                counts.get(),
                1,
                mpi_type<zisa::int_t>(),
                comm);
  for (int num_ranks = size; num_ranks > 1;
       num_ranks = zisa::div_up(num_ranks, 2)) {
    const int num_recv = zisa::div_up(num_ranks, 2);
    for (zisa::int_t offset = 0; offset < Mk_.size(); offset += BUFFER_SIZE) {
      const zisa::int_t num_elements
          = zisa::min(BUFFER_SIZE, Mk_.size() - offset);
      if (rank < num_recv) {
        if (rank + num_recv < num_ranks) {
          const int src = rank + num_recv;
          fmt::print("Combining {} -> {}\n", src, rank);
          MPI_Recv(buffer_Mk.get(),
                   num_elements,
                   mpi_type<real_t>(),
                   src,
                   0,
                   comm,
                   MPI_STATUS_IGNORE);
          MPI_Recv(buffer_Sk.get(),
                   num_elements,
                   mpi_type<real_t>(),
                   src,
                   1,
                   comm,
                   MPI_STATUS_IGNORE);
          const zisa::int_t nA = counts[rank];
          const zisa::int_t nB = counts[src];
          const zisa::int_t nAB = nA + nB;
          counts[rank] = nAB;
          count_ = nAB;
          for (size_t i = 0; i < num_elements; ++i) {
            const real_t MkA = Mk_[offset + i];
            const real_t MkB = buffer_Mk[i];
            const real_t SkA = Sk_[offset + i];
            const real_t SkB = buffer_Sk[i];
            const real_t delta = MkB - MkA;
            const real_t MkAB = (nA * MkA + nB * MkB) / nAB;
            const real_t SkAB = SkA + SkB + (delta * delta * nA * nB) / nAB;
            Mk_[offset + i] = MkAB;
            Sk_[offset + i] = SkAB;
          }
        }
      } else if (rank < num_ranks) {
        const int dst = rank - num_recv;
        MPI_Send(
            Mk_.raw() + offset, num_elements, mpi_type<real_t>(), dst, 0, comm);
        MPI_Send(
            Sk_.raw() + offset, num_elements, mpi_type<real_t>(), dst, 1, comm);
      }
    }
  }
  if (rank == 0) {
    finalize();
  }
}
#endif

template <int Dim>
zisa::array_const_view<real_t, Dim + 1> StatisticsRecorder<Dim>::mean() const {
  LOG_ERR_IF(
      !is_finalized_,
      "Please call StatisticsRecorder.finalize() before accessing the mean");
  return Mk_;
}

template <int Dim>
zisa::array_const_view<real_t, Dim + 1>
StatisticsRecorder<Dim>::variance() const {
  LOG_ERR_IF(!is_finalized_,
             "Please call StatisticsRecorder.finalize() before accessing the "
             "variance");
  return Sk_;
}

template class StatisticsRecorder<1>;
template class StatisticsRecorder<2>;
template class StatisticsRecorder<3>;

}
