/*
 * This file is part of azeban (https://github.com/TobiasRohner/azeban).
 * Copyright (c) 2021 Tobias Rohner.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef MPI_TYPES_H_
#define MPI_TYPES_H_

#include <azeban/complex.hpp>
#include <mpi.h>

namespace azeban {

namespace internal {

template <typename T>
struct MPI_Type {
  static_assert(!std::is_same_v<T, T>, "Unknown MPI Datatype");
};

}

template <typename T>
MPI_Datatype mpi_type() {
  static internal::MPI_Type<T> type = internal::MPI_Type<T>();
  return type.type;
}

template <typename T>
MPI_Datatype mpi_type(T) {
  return mpi_type<T>();
}

namespace internal {

template <>
struct MPI_Type<int8_t> {
  MPI_Datatype type = MPI_INT8_T;
};

template <>
struct MPI_Type<uint8_t> {
  MPI_Datatype type = MPI_UINT8_T;
};

template <>
struct MPI_Type<int16_t> {
  MPI_Datatype type = MPI_INT16_T;
};

template <>
struct MPI_Type<uint16_t> {
  MPI_Datatype type = MPI_UINT16_T;
};

template <>
struct MPI_Type<int32_t> {
  MPI_Datatype type = MPI_INT32_T;
};

template <>
struct MPI_Type<uint32_t> {
  MPI_Datatype type = MPI_UINT32_T;
};

template <>
struct MPI_Type<int64_t> {
  MPI_Datatype type = MPI_INT64_T;
};

template <>
struct MPI_Type<uint64_t> {
  MPI_Datatype type = MPI_UINT64_T;
};

template <>
struct MPI_Type<float> {
  MPI_Datatype type = MPI_FLOAT;
};

template <>
struct MPI_Type<double> {
  MPI_Datatype type = MPI_DOUBLE;
};

template <typename T>
struct MPI_Type<Complex<T>> {
  MPI_Type() {
    MPI_Type_contiguous(2, mpi_type<T>(), &type);
    MPI_Type_commit(&type);
  }

  ~MPI_Type() {
    // Not allowed to free the type because this destructor will be called after
    // MPI_Finalize() MPI_Type_free(&type);
  }

  MPI_Datatype type;
};

}

}

#endif
