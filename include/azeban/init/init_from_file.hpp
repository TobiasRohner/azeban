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
#ifndef INIT_FROM_FILE_HPP_
#define INIT_FROM_FILE_HPP_

#include "initializer.hpp"
#include <vector>

namespace azeban {

template <int Dim>
class InitFromFile : public Initializer<Dim> {
  using super = Initializer<Dim>;

public:
  InitFromFile(const std::string &experiment,
               const std::string &time,
               zisa::int_t sample_idx_start = 0)
      : sample_(sample_idx_start), experiment_(experiment), time_(time) {}
  InitFromFile(const InitFromFile &) = default;
  InitFromFile(InitFromFile &&) = default;

  virtual ~InitFromFile() override = default;

  InitFromFile &operator=(const InitFromFile &) = default;
  InitFromFile &operator=(InitFromFile &&) = default;

protected:
  virtual void
  do_initialize(const zisa::array_view<real_t, Dim + 1> &u) override;
  virtual void
  do_initialize(const zisa::array_view<complex_t, Dim + 1> &u_hat) override;

private:
  zisa::int_t sample_;
  std::string experiment_;
  std::string time_;

  std::string filename() const;
  std::vector<int> get_varids(int ncid) const;
  std::vector<std::string> get_varnames(int ncid) const;
  std::vector<size_t> get_dims(int ncid, int varid) const;
  void read_component(int ncid,
                      const std::string &name,
                      const zisa::array_view<real_t, Dim> &u) const;
  void read_u(int ncid, const zisa::array_view<real_t, Dim + 1> &u) const;
  void read_u_hat(int ncid,
                  const zisa::array_view<complex_t, Dim + 1> &u_hat) const;
  void read_rho(int ncid, const zisa::array_view<real_t, Dim + 1> &rho) const;
  void read_rho_hat(int ncid,
                  const zisa::array_view<complex_t, Dim + 1> &rho_hat) const;
  void read_omega(int ncid,
                  const zisa::array_view<real_t, Dim + 1> &omega) const;
  void
  read_omega_hat(int ncid,
                 const zisa::array_view<complex_t, Dim + 1> &omega_hat) const;
  void init(int ncid, const zisa::array_view<real_t, Dim + 1> &u) const;
  void init(int ncid, const zisa::array_view<complex_t, Dim + 1> &u_hat) const;
};

}

#endif
