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
#ifndef FFT_FACTORY_H_
#define FFT_FACTORY_H_

#include <azeban/config.hpp>
#include <azeban/operations/fft.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

template <int Dim>
std::shared_ptr<FFT<Dim>>
make_fft(const zisa::array_view<complex_t, Dim + 1> &u_hat,
         const zisa::array_view<real_t, Dim + 1> &u,
         int direction = FFT_FORWARD | FFT_BACKWARD);

}

#endif
