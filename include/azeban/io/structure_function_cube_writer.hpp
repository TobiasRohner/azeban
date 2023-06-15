#ifndef AZEBAN_IO_STRUCTURE_FUNCTION_CUBE_WRITER_HPP_
#define AZEBAN_IO_STRUCTURE_FUNCTION_CUBE_WRITER_HPP_

#include <azeban/io/structure_function_writer.hpp>
#include <azeban/operations/structure_function_functionals.hpp>

namespace azeban {

template <int Dim>
class StructureFunctionCubeWriter
    : public StructureFunctionWriter<Dim, SFCubeFunctional> {
  using super = StructureFunctionWriter<Dim, SFCubeFunctional>;

public:
  ANY_DEVICE
  StructureFunctionCubeWriter(const std::string &path,
                              const Grid<Dim> &grid,
                              const std::vector<real_t> &snapshot_times,
                              zisa::int_t sample_idx_start,
                              real_t p,
                              ssize_t max_h);
  ANY_DEVICE_INLINE
  StructureFunctionCubeWriter(const StructureFunctionCubeWriter &)
      = default;
  ANY_DEVICE_INLINE StructureFunctionCubeWriter(StructureFunctionCubeWriter &&)
      = default;

  virtual ~StructureFunctionCubeWriter() override = default;

  ANY_DEVICE_INLINE StructureFunctionCubeWriter &
  operator=(const StructureFunctionCubeWriter &)
      = default;
  ANY_DEVICE_INLINE StructureFunctionCubeWriter &
  operator=(StructureFunctionCubeWriter &&)
      = default;

  using super::next_timestep;
  using super::reset;
  using super::write;
};

}

#endif
