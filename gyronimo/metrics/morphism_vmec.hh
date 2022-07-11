// ::gyronimo:: - gyromotion for the people, by the people -
// An object-oriented library for gyromotion applications in plasma physics.
// Copyright (C) 2022 Paulo Rodrigues.

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

// @morphism_vmec.hh, this file is part of ::gyronimo::

#ifndef GYRONIMO_MORPHISM_VMEC
#define GYRONIMO_MORPHISM_VMEC

#include <gyronimo/metrics/morphism.hh>
#include <gyronimo/parsers/parser_vmec.hh>
#include <gyronimo/interpolators/interpolator1d.hh>

namespace gyronimo {

//! Morphism from `vmec` curvilinear coordinates.
class morphism_vmec : public morphism {
 typedef std::valarray<double> narray_type;
 public:
  morphism_vmec(
      const parser_vmec *parser, const interpolator1d_factory *ifactory);
  virtual ~morphism_vmec() override;
  virtual IR3 operator()(const IR3& q) const override;
  virtual IR3 inverse(const IR3& x) const override;
  virtual dIR3 del(const IR3& q) const override;
  const parser_vmec* parser() const {return parser_;};
 private:
  const parser_vmec *parser_;
  double b0_;
  int mnmax_, mnmax_nyq_, ns_, mpol_, ntor_, nfp_, signsgs_; 
  narray_type xm_, xn_, xm_nyq_, xn_nyq_; 
  interpolator1d **Rmnc_;
  interpolator1d **Zmns_;
  static double reduce_2pi(double x);
  static std::tuple<double, double> reflection_past_axis(double s, double theta);
};

}

#endif // GYRONIMO_MORPHISM_VMEC



