#include <azeban/io/writer_collection.hpp>

namespace azeban {

template <int Dim>
WriterCollection<Dim>::WriterCollection(const Grid<Dim> &grid)
    : super(grid, std::vector<double>()) {}

template <int Dim>
void WriterCollection<Dim>::add_writer(std::unique_ptr<Writer<Dim>> &&writer) {
  writers_.push_back(std::move(writer));
}

template <int Dim>
void WriterCollection<Dim>::reset() {
  for (auto &writer : writers_) {
    writer->reset();
  }
}

template <int Dim>
double WriterCollection<Dim>::next_timestep() const {
  double t = writers_[0]->next_timestep();
  for (size_t i = 1; i < writers_.size(); ++i) {
    t = zisa::min(t, writers_[i]->next_timestep());
  }
  return t;
}

template <int Dim>
void WriterCollection<Dim>::write(
    const zisa::array_const_view<real_t, Dim + 1> &u, double t) {
  for (auto &writer : writers_) {
    if (writer->next_timestep() == t) {
      writer->write(u, t);
    }
  }
}

template <int Dim>
void WriterCollection<Dim>::write(
    const zisa::array_const_view<complex_t, Dim + 1> &u_hat, double t) {
  for (auto &writer : writers_) {
    if (writer->next_timestep() == t) {
      writer->write(u_hat, t);
    }
  }
}

#if AZEBAN_HAS_MPI
template <int Dim>
void WriterCollection<Dim>::write(
    const zisa::array_const_view<real_t, Dim + 1> &u,
    double t,
    const Communicator *comm) {
  for (auto &writer : writers_) {
    if (writer->next_timestep() == t) {
      writer->write(u, t, comm);
    }
  }
}

template <int Dim>
void WriterCollection<Dim>::write(
    const zisa::array_const_view<complex_t, Dim + 1> &u_hat,
    double t,
    const Communicator *comm) {
  for (auto &writer : writers_) {
    if (writer->next_timestep() == t) {
      writer->write(u_hat, t, comm);
    }
  }
}
#endif

template class WriterCollection<1>;
template class WriterCollection<2>;
template class WriterCollection<3>;

}
