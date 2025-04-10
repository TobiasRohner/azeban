#ifndef AZEBAN_IO_ABSOLUTE_LONGITUDINAL_STRUCTURE_FUNCTION_WRITER_HPP_
#define AZEBAN_IO_ABSOLUTE_LONGITUDINAL_STRUCTURE_FUNCTION_WRITER_HPP_

#include <azeban/io/structure_function_writer.hpp>
#include <azeban/operations/structure_function_functionals.hpp>

namespace azeban {

template <int Dim>
class AbsoluteLongitudinalStructureFunctionWriter
    : public StructureFunctionWriter<Dim, SFAbsoluteLongitudinalFunctional> {
  using super = StructureFunctionWriter<Dim, SFAbsoluteLongitudinalFunctional>;

public:
  ANY_DEVICE
  AbsoluteLongitudinalStructureFunctionWriter(
      const std::string &path,
      const Grid<Dim> &grid,
      const std::vector<double> &snapshot_times,
      zisa::int_t sample_idx_start,
      real_t p,
      ssize_t max_h);
  ANY_DEVICE_INLINE
  AbsoluteLongitudinalStructureFunctionWriter(
      const AbsoluteLongitudinalStructureFunctionWriter &)
      = default;
  ANY_DEVICE_INLINE
  AbsoluteLongitudinalStructureFunctionWriter(
      AbsoluteLongitudinalStructureFunctionWriter &&)
      = default;

  virtual ~AbsoluteLongitudinalStructureFunctionWriter() override = default;

  ANY_DEVICE_INLINE AbsoluteLongitudinalStructureFunctionWriter &
  operator=(const AbsoluteLongitudinalStructureFunctionWriter &)
      = default;
  ANY_DEVICE_INLINE AbsoluteLongitudinalStructureFunctionWriter &
  operator=(AbsoluteLongitudinalStructureFunctionWriter &&)
      = default;

  using super::next_timestep;
  using super::reset;
  using super::write;
};

}

#endif
