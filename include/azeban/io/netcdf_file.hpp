#ifndef AZEBAN_IO_NETCDF_FILE_HPP_
#define AZEBAN_IO_NETCDF_FILE_HPP_

#include <azeban/grid.hpp>
#include <azeban/io/netcdf_writer.hpp>
#include <azeban/io/writer.hpp>
#include <string>
#include <vector>

namespace azeban {

template <int Dim>
class NetCDFFile : public Writer<Dim> {
  using super = Writer<Dim>;

public:
  NetCDFFile(const std::string &path,
             const Grid<Dim> &grid,
             zisa::int_t num_samples,
             zisa::int_t sample_idx_start,
             const std::string &config,
             const std::string &script);
  NetCDFFile(const NetCDFFile &) = delete;
  NetCDFFile(NetCDFFile &&) = default;

  virtual ~NetCDFFile() override;

  NetCDFFile &operator=(const NetCDFFile &) = delete;
  NetCDFFile &operator=(NetCDFFile &&) = default;

  int ncid() const;

  void add_writer(std::unique_ptr<NetCDFWriter<Dim>> &&writer);

  virtual void reset() override;
  virtual double next_timestep() const override;
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

private:
  int ncid_ = 0;
  int dimid_member_;
  int dimid_dim_[Dim];
  std::vector<std::unique_ptr<NetCDFWriter<Dim>>> writers_;
};

}

#endif
