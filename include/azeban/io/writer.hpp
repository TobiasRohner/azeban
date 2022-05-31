#ifndef AZEBAN_IO_WRITER_HPP_
#define AZEBAN_IO_WRITER_HPP_

#include <azeban/config.hpp>
#include <zisa/memory/array_view.hpp>
#if AZEBAN_HAS_MPI
#include <azeban/mpi/communicator.hpp>
#endif

namespace azeban {

template <int Dim>
class Writer {
public:
  virtual ~Writer() = default;

  virtual void reset() = 0;
  virtual real_t next_timestep() const = 0;
  virtual void write(const zisa::array_const_view<real_t, Dim + 1> &u, real_t t)
      = 0;
  virtual void write(const zisa::array_const_view<complex_t, Dim + 1> &u_hat,
                     real_t t)
      = 0;
#if AZEBAN_HAS_MPI
  virtual void write(const zisa::array_const_view<real_t, Dim + 1> &u,
                     real_t t,
                     const Communicator *comm)
      = 0;
  virtual void write(const zisa::array_const_view<complex_t, Dim + 1> &u_hat,
                     real_t t,
                     const Communicator *comm)
      = 0;
#endif

protected:
};

}

#endif
