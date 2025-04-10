#ifndef AZEBAN_IO_WRITER_COLLECTION_HPP_
#define AZEBAN_IO_WRITER_COLLECTION_HPP_

#include <azeban/io/writer.hpp>
#include <memory>
#include <vector>

namespace azeban {

template <int Dim>
class WriterCollection : public Writer<Dim> {
  using super = Writer<Dim>;

public:
  WriterCollection(const Grid<Dim> &grid);
  WriterCollection(const WriterCollection &) = default;
  WriterCollection(WriterCollection &&) = default;
  WriterCollection &operator=(const WriterCollection &) = default;
  WriterCollection &operator=(WriterCollection &&) = default;
  virtual ~WriterCollection() override = default;

  void add_writer(std::unique_ptr<Writer<Dim>> &&writer);

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
  std::vector<std::unique_ptr<Writer<Dim>>> writers_;
};

}

#endif
