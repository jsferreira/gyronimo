// ::gyronimo:: - gyromotion for the people, by the people -
// An object-oriented library for gyromotion applications in plasma physics.
// Copyright (C) 2021 Paulo Rodrigues.

// ::gyronimo:: is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// ::gyronimo:: is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with ::gyronimo::.  If not, see <https://www.gnu.org/licenses/>.

// @linspace.hh, this file is part of ::gyronimo::

#ifndef GYRONIMO_LINSPACE
#define GYRONIMO_LINSPACE

// #include <ranges>
#include <range/v3/all.hpp>
#include <algorithm>
#include <gyronimo/core/error.hh>

namespace gyronimo {

//! Returns a `Container` of evenly spaced samples, similar to numpy's linspace.
template<typename Container> requires
  ranges::sized_range<Container> &&
  std::constructible_from<Container, size_t> &&
  std::floating_point<typename Container::value_type>
inline
Container linspace(
    const typename Container::value_type& start,
    const typename Container::value_type& end,
    size_t number) {
  if(end <= start || number < 2) error(
      __func__, __FILE__, __LINE__, "inconsistent arguments.", 1);
  Container samples(number);
  double delta = (end - start)/(number - 1);
  ranges::transform(
      ranges::views::iota(0u, number), ranges::begin(samples),
      [start, delta](size_t i) {return start + i*delta;});
  return samples;
}

} // end namespace gyronimo.

#endif // GYRONIMO_LINSPACE
