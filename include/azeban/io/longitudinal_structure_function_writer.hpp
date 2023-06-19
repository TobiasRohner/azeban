#ifndef AZEBAN_IO_LONGITUDINAL_STRUCTURE_FUNCTION_WRITER_HPP_
#define AZEBAN_IO_LONGITUDINAL_STRUCTURE_FUNCTION_WRITER_HPP_

#include <azeban/io/structure_function_writer.hpp>
#include <azeban/operations/structure_function_functionals.hpp>

namespace azeban {

template <int Dim>
class LongitudinalStructureFunctionWriter
    : public StructureFunctionWriter<Dim, SFLongitudinalFunctional> {
  using super = StructureFunctionWriter<Dim, SFLongitudinalFunctional>;

public:
  ANY_DEVICE
  LongitudinalStructureFunctionWriter(const std::string &path,
                                      const Grid<Dim> &grid,
                                      const std::vector<real_t> &snapshot_times,
                                      zisa::int_t sample_idx_start,
                                      real_t p,
                                      ssize_t max_h);
  ANY_DEVICE_INLINE
  LongitudinalStructureFunctionWriter(
      const LongitudinalStructureFunctionWriter &)
      = default;
  ANY_DEVICE_INLINE
  LongitudinalStructureFunctionWriter(LongitudinalStructureFunctionWriter &&)
      = default;

  virtual ~LongitudinalStructureFunctionWriter() override = default;

  ANY_DEVICE_INLINE LongitudinalStructureFunctionWriter &
  operator=(const LongitudinalStructureFunctionWriter &)
      = default;
  ANY_DEVICE_INLINE LongitudinalStructureFunctionWriter &
  operator=(LongitudinalStructureFunctionWriter &&)
      = default;

  using super::next_timestep;
  using super::reset;
  using super::write;
};

}

#endif
