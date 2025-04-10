#ifndef AZEBAN_IO_ENERGY_SPECTRUM_WRITER_HPP_
#define AZEBAN_IO_ENERGY_SPECTRUM_WRITER_HPP_

#include <azeban/io/writer.hpp>
#include <fstream>
#include <string>
#if AZEBAN_HAS_MPI
#include <mpi.h>
#endif

namespace azeban {

template <int Dim>
class EnergySpectrumWriter : public Writer<Dim> {
  using super = Writer<Dim>;

public:
  EnergySpectrumWriter(const std::string &path,
                       const Grid<Dim> &grid,
                       const std::vector<double> &snapshot_times,
                       int sample_idx_start);
#if AZEBAN_HAS_MPI
  EnergySpectrumWriter(const std::string &path,
                       const Grid<Dim> &grid,
                       const std::vector<double> &snapshot_times,
                       int sample_idx_start,
                       const Communicator *comm);
#endif
  EnergySpectrumWriter(const EnergySpectrumWriter &) = default;
  EnergySpectrumWriter &operator=(const EnergySpectrumWriter &) = default;

  virtual ~EnergySpectrumWriter() override = default;

  using super::next_timestep;
  virtual void reset() override;
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
  std::ofstream file_;
#if AZEBAN_HAS_MPI
  MPI_Comm samples_comm_;
#endif
};

}

#endif
