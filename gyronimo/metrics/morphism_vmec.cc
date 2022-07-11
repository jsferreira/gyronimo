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

// @morphism_vmec.cc, this file is part of ::gyronimo::

#include <numbers>
#include <gyronimo/core/error.hh>
#include <gyronimo/core/multiroot.hh>
#include <gyronimo/metrics/morphism_vmec.hh>

namespace gyronimo{

morphism_vmec::morphism_vmec(
   const parser_vmec *p, const interpolator1d_factory *ifactory) 
    : parser_(p), b0_(p->B_0()), mnmax_(p->mnmax()), mnmax_nyq_(p->mnmax_nyq()),
      ns_(p->ns()), mpol_(p->mpol()), ntor_(p->ntor()), 
      signsgs_(p->signgs()), nfp_(p->nfp()),
      xm_(p->xm()), xn_(p->xn()), xm_nyq_(p->xm_nyq()), xn_nyq_(p->xn_nyq()),
      Rmnc_(nullptr), Zmns_(nullptr)
      {
      // set radial grid block
      dblock_adapter s_range(p->radius());
      dblock_adapter s_half_range(p->radius_half());
      // set spectral components 
      Rmnc_ = new interpolator1d* [xm_.size()];
      Zmns_ = new interpolator1d* [xm_.size()];
      //@todo NEED TO FIX AXIS AND EDGE! TBI! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      #pragma omp parallel for
      for(size_t i=0; i<xm_.size(); i++) {
        std::slice s_cut (i, s_range.size(), xm_.size());
        std::valarray<double> rmnc_i = (p->rmnc())[s_cut];
        Rmnc_[i] = ifactory->interpolate_data( s_range, dblock_adapter(rmnc_i));
        std::valarray<double> zmnc_i = (p->zmns())[s_cut];
        Zmns_[i] = ifactory->interpolate_data( s_range, dblock_adapter(zmnc_i));
      };
}
morphism_vmec::~morphism_vmec() {
  if(Rmnc_) delete Rmnc_;
  if(Zmns_) delete Zmns_;
}
IR3 morphism_vmec::operator()(const IR3& q) const {
  double s = q[IR3::u], zeta = q[IR3::v], theta = q[IR3::w];
  double R = 0.0; double z = 0.0;
  #pragma omp parallel for reduction(+: R, Z)
  for (size_t i = 0; i<xm_.size(); i++) {  
    double m = xm_[i]; double n = xn_[i];
    double cosmn = std::cos( m*theta - n*zeta );
    double sinmn = std::sin( m*theta - n*zeta );
     // assuming for now that vmec equilibrium has stellarator symmetry.
    R += (*Rmnc_[i])(s) * cosmn; 
    z += (*Zmns_[i])(s) * sinmn;
  };
  return {R*std::cos(zeta), R*std::sin(zeta), z};
}
IR3 morphism_vmec::inverse(const IR3& X) const {
  typedef std::array<double, 2> IR2;
  double x = X[IR3::u], y = X[IR3::v], z = X[IR3::w];
  double R = std::sqrt(x*x + y*y); double zeta = std::atan2(y, x);
  std::function<IR2(const IR2&)> zero_function =
      [&](const IR2& args) {
        auto [s, theta] = reflection_past_axis(args[0], args[1]);
        double R_ = 0.0; double z_ = 0.0;
        #pragma omp parallel for reduction(+: R, Z)
        for (size_t i = 0; i<xm_.size(); i++) {  
          double m = xm_[i]; double n = xn_[i];
          double cosmn = std::cos( m*theta - n*zeta );
          double sinmn = std::sin( m*theta - n*zeta );
          // assuming for now that vmec equilibrium has stellarator symmetry.
          R_ += (*Rmnc_[i])(s) * cosmn; 
          z_ += (*Zmns_[i])(s) * sinmn;
        };
        return IR2({R_ - R, z_ - z});
      };
  IR2 guess = {0.5, std::atan2(z, R - parser_->R_0())};
  auto root_finder = multiroot(1.0e-15, 100);
  IR2 roots = root_finder(zero_function, guess);
  auto [s, theta] = reflection_past_axis(roots[0], roots[1]);
  return {s, zeta, theta};
}
dIR3 morphism_vmec::del(const IR3& q) const {
  double s = q[IR3::u], zeta = q[IR3::v], theta = q[IR3::w];
  double R = 0.0, Z = 0.0;
  double dR_ds = 0.0, dR_dtheta = 0.0, dR_dzeta = 0.0;
  double dZ_ds = 0.0, dZ_dtheta = 0.0, dZ_dzeta = 0.0;
  #pragma omp parallel for reduction(+: R, dR_ds, dR_dtheta, dR_dzeta, d2R_ds2, d2R_dsdtheta, d2R_dsdzeta, d2R_dtheta2, d2R_dthetadzeta, d2R_dzeta2, Z, dZ_ds ,dZ_dtheta, dZ_dzeta, d2Z_ds2, d2Z_dsdtheta, d2Z_dsdzeta, d2Z_dtheta2, d2Z_dthetadzeta, d2Z_dzeta2)
  for (size_t i = 0; i<xm_.size(); i++) {  
    double m = xm_[i]; double n = xn_[i];
    double cosmn = std::cos( m*theta - n*zeta );
    double sinmn = std::sin( m*theta - n*zeta );
    double rmnc_i = (*Rmnc_[i])(s); 
    double zmns_i = (*Zmns_[i])(s);
    double d_rmnc_i = (*Rmnc_[i]).derivative(s); 
    double d_zmns_i = (*Zmns_[i]).derivative(s); 
    // assuming for now that vmec equilibrium has stellarator symmetry.
    R += rmnc_i * cosmn; Z += zmns_i * sinmn;
    dR_ds += d_rmnc_i * cosmn; 
    dR_dtheta -= m * rmnc_i * sinmn; 
    dR_dzeta += n * rmnc_i * sinmn;
    dZ_ds += d_zmns_i * sinmn; 
    dZ_dtheta += m * zmns_i * cosmn; 
    dZ_dzeta -= n * zmns_i * cosmn; 
}
  double cos = std::cos(zeta), sin = std::sin(zeta);
  return {dR_ds * cos, dR_dzeta * cos - R*sin, dR_dtheta * cos, 
          dR_ds * sin, dR_dzeta * sin + R*cos, dR_dtheta * sin,
          dZ_ds, dZ_dzeta, dZ_dtheta};
}
std::tuple<double, double> morphism_vmec::reflection_past_axis(
    double s, double theta) {
  if(s < 0)
    return {-s, reduce_2pi(theta + std::numbers::pi)};
  else
    return {s, theta};
}
double morphism_vmec::reduce_2pi(double x) {
  double l = 2*std::numbers::pi;
  return (x -= l*std::floor(x/l));
}

} // end namespace gyronimo
