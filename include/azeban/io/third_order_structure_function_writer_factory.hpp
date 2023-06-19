#ifndef AZEBAN_IO_THIRD_ORDER_STRUCTURE_FUNCTION_WRITER_FACTORY_HPP_
#define AZEBAN_IO_THIRD_ORDER_STRUCTURE_FUNCTION_WRITER_FACTORY_HPP_

#include <azeban/io/writer.hpp>
#include <memory>
#include <nlohmann/json.hpp>

namespace azeban {

template <int Dim>
std::unique_ptr<Writer<Dim>>
make_third_order_structure_function_writer(const nlohmann::json &config,
                                           const Grid<Dim> &grid,
                                           zisa::int_t sample_idx_start);

}

#endif
