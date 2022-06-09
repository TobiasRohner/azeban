#ifndef AZEBAN_IO_WRITER_FACTORY_HPP_
#define AZEBAN_IO_WRITER_FACTORY_HPP_

#include <azeban/grid.hpp>
#include <azeban/io/writer.hpp>
#include <nlohmann/json.hpp>

namespace azeban {

template <int Dim>
std::unique_ptr<Writer<Dim>> make_writer(const nlohmann::json &config,
                                         const Grid<Dim> &grid,
                                         zisa::int_t sample_idx_start,
                                         void *work_area = nullptr);

}

#endif
