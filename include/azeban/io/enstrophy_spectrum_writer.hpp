#ifndef AZEBAN_IO_ENSTROPHY_SPECTRUM_WRITER_HPP_
#define AZEBAN_IO_ENSTROPHY_SPECTRUM_WRITER_HPP_

#include <azeban/io/writer.hpp>
#include <fstream>
#include <string>
#if AZEBAN_HAS_MPI
#include <mpi.h>
#endif

namespace azeban {

template <int Dim>
class EnstrophySpectrumWriter : public Writer<Dim> {
  using super = Writer<Dim>;

public:
  EnstrophySpectrumWriter(const std::string &path,
                          const Grid<Dim> &grid,
                          const std::vector<real_t> &snapshot_times,
                          int sample_idx_start);
#if AZEBAN_HAS_MPI
  EnstrophySpectrumWriter(const std::string &path,
                          const Grid<Dim> &grid,
                          const std::vector<real_t> &snapshot_times,
                          int sample_idx_start,
                          const Communicator *comm);
#endif
  EnstrophySpectrumWriter(const EnstrophySpectrumWriter &) = default;
  EnstrophySpectrumWriter &operator=(const EnstrophySpectrumWriter &) = default;

  virtual ~EnstrophySpectrumWriter() override = default;

  using super::next_timestep;
  virtual void reset() override;
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
  std::ofstream file_;
#if AZEBAN_HAS_MPI
  MPI_Comm samples_comm_;
#endif
};

}

#endif
