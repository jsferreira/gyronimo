// ::gyronimo:: - gyromotion for the people, by the people -
// An object-oriented library for gyromotion applications in plasma physics.
// Copyright (C) 2022-2023 Jorge Ferreira and Paulo Rodrigues.

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

// @metric_vmec.cc, this file is part of ::gyronimo::

#include <gyronimo/metrics/metric_vmec.hh>
#include <unistd.h>
#include <numbers>

namespace gyronimo{

metric_vmec::metric_vmec(
    const parser_vmec *p, const interpolator1d_factory *ifactory, const bool cached = true ) 
    : parser_(p), b0_(p->B_0()), mnmax_(p->mnmax()), mnmax_nyq_(p->mnmax_nyq()),
      ns_(p->ns()), mpol_(p->mpol()), ntor_(p->ntor()), 
      signsgs_(p->signgs()), nfp_(p->nfp()),
      xm_(p->xm()), xn_(p->xn()), xm_nyq_(p->xm_nyq()), xn_nyq_(p->xn_nyq()),
      Rmnc_(nullptr), Zmns_(nullptr), gmnc_(nullptr), ds_half_cell_(0.5/(p->ns()-1.0)),
      cached_operator_position_({0.0,0.0,0.0}), cached_operator_value_({0.0,0.0,0.0,0.0,0.0,0.0}),
      cached_del_position_({0.0,0.0,0.0}), cached_del_value_({0.0,0.0,0.0,0.0,0.0,0.0}),
      use_cache_(cached)
      {
    // set radial grid block
    dblock_adapter s_range(p->radius());
    dblock_adapter s_half_range(p->radius_half());
    // set spectral components 
    Rmnc_ = new interpolator1d* [xm_.size()];
    Zmns_ = new interpolator1d* [xm_.size()];
    gmnc_ = new interpolator1d* [xm_nyq_.size()];
//@todo NEED TO FIX AXIS AND EDGE! TBI! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    #pragma omp parallel for
    for(size_t i=0; i<xm_.size(); i++) {
      std::slice s_cut (i, s_range.size(), xm_.size());
      std::valarray<double> rmnc_i = (p->rmnc())[s_cut];
      Rmnc_[i] = ifactory->interpolate_data( s_range, dblock_adapter(rmnc_i));
      // note that theta_g = - theta_vmec -> Z_g = - Z_vmec 
      std::valarray<double> zmnc_i = - (p->zmns())[s_cut];
      Zmns_[i] = ifactory->interpolate_data( s_range, dblock_adapter(zmnc_i));
    };
    #pragma omp parallel for
    for(size_t i=0; i<xm_nyq_.size(); i++) {
      // note that gmnc is defined at half mesh
      std::slice s_h_cut (i+xm_nyq_.size(), s_half_range.size(), xm_nyq_.size());
      // vmec in gyronimo is right-handed
      std::valarray<double> gmnc_i = - (p->gmnc())[s_h_cut];
      gmnc_[i] = ifactory->interpolate_data( s_half_range, dblock_adapter(gmnc_i));
    };
}
metric_vmec::~metric_vmec() {
  if(Rmnc_) delete Rmnc_;
  if(Zmns_) delete Zmns_;
  if(gmnc_) delete gmnc_;
}
auto metric_vmec::operator()(const IR3& position) const -> SM3  {
  double s = position[IR3::u];
  double theta = position[IR3::v];
  double zeta = position[IR3::w];
  if (use_cache_) {
    IR3 delta = (position - cached_operator_position_); delta = delta*delta;
    double sdelta = delta[0]+delta[1]+delta[2];
    if (sdelta < 1.0e-32) return cached_operator_value_; // criteria for equalness hardwired for now
  }
  if (s < 0.0) {
    s = -s;
    theta = theta + std::numbers::pi; // @todo, discuss clockwise versus anti-clockwise rotating of theta when mirroring @ s=0
  }
  else if (s > 1) s = 1;
  double R = 0.0, dR_ds = 0.0, dR_dtheta = 0.0, dR_dzeta = 0.0;
  double Z = 0.0, dZ_ds = 0.0, dZ_dtheta = 0.0, dZ_dzeta = 0.0;
  #pragma omp parallel for reduction(+: R, Z, dR_ds, dR_dtheta, dR_dzeta, dZ_ds, dZ_dtheta, dZ_dzeta)
  for (size_t i = 0; i<xm_.size(); i++) {  
    double m = xm_[i]; double n = xn_[i];
    double cosmn = std::cos( m*theta + n*zeta );
    double sinmn = std::sin( m*theta + n*zeta );
    double rmnc_i = (*Rmnc_[i])(s);
    double zmns_i = (*Zmns_[i])(s);
    // assuming for now that vmec equilibrium has stellarator symmetry.
    R += rmnc_i * cosmn; 
    Z += zmns_i * sinmn;
    dR_ds += (*Rmnc_[i]).derivative(s) * cosmn; 
    dR_dtheta -= m * rmnc_i * sinmn; 
    dR_dzeta -= n * rmnc_i * sinmn;
    dZ_ds += (*Zmns_[i]).derivative(s) * sinmn; 
    dZ_dtheta += m * zmns_i * cosmn; 
    dZ_dzeta += n * zmns_i * cosmn; 
  };
  SM3 value = {
    dR_ds * dR_ds + dZ_ds * dZ_ds,                      // g_uu
    dR_ds * dR_dtheta + dZ_ds * dZ_dtheta,              // g_uv
    dR_ds * dR_dzeta + dZ_ds * dZ_dzeta,                // g_uw
    dR_dtheta * dR_dtheta + dZ_dtheta * dZ_dtheta,      // g_vv
    dR_dtheta * dR_dzeta + dZ_dtheta * dZ_dzeta,        // g_vw
    R * R + dR_dzeta * dR_dzeta + dZ_dzeta * dZ_dzeta   // g_ww
  };
  if (use_cache_) {
    cached_operator_position_ = position;
    cached_operator_value_ = value;
  } ;
  return value;
}
auto metric_vmec::del(const IR3& position) const -> dSM3 {
  double s = position[IR3::u];
  double theta = position[IR3::v];
  double zeta = position[IR3::w];
  if (use_cache_) {
    IR3 delta = (position - cached_del_position_); delta = delta*delta;
    double sdelta = delta[0]+delta[1]+delta[2];
    if (sdelta < 1.0e-32) return cached_del_value_; // criteria for equalness hardwired for now
  };
  if (s < 0.0) {
    s = -s;
    theta = theta + std::numbers::pi; // @todo, discuss clockwise versus anti-clockwise rotating of theta when mirroring @ s=0
  }
  else if (s > 1) s = 1;

  double R = 0.0, Z = 0.0;
  double dR_ds = 0.0,        dR_dtheta = 0.0,       dR_dzeta = 0.0;
  double d2R_ds2 = 0.0,      d2R_dsdtheta = 0.0,    d2R_dsdzeta = 0.0;
  double d2R_dthetads = 0.0, d2R_dtheta2 = 0.0,     d2R_dthetadzeta = 0.0; 
  double d2R_dzetads = 0.0,  d2R_dzetadtheta = 0.0, d2R_dzeta2 = 0.0;
  double dZ_ds = 0.0,        dZ_dtheta = 0.0,       dZ_dzeta = 0.0;
  double d2Z_ds2 = 0.0,      d2Z_dsdtheta = 0.0,    d2Z_dsdzeta = 0.0;
  double d2Z_dthetads = 0.0, d2Z_dtheta2 = 0.0,     d2Z_dthetadzeta = 0.0; 
  double d2Z_dzetads = 0.0,  d2Z_dzetadtheta = 0.0, d2Z_dzeta2 = 0.0;

  #pragma omp parallel for reduction(+: R, dR_ds, dR_dtheta, dR_dzeta, d2R_ds2, d2R_dsdtheta, d2R_dsdzeta, d2R_dtheta2, d2R_dthetadzeta, d2R_dzeta2, Z, dZ_ds ,dZ_dtheta, dZ_dzeta, d2Z_ds2, d2Z_dsdtheta, d2Z_dsdzeta, d2Z_dtheta2, d2Z_dthetadzeta, d2Z_dzeta2)
  for (size_t i = 0; i<xm_.size(); i++) {  
    double m = xm_[i]; double n = xn_[i];
    double cosmn = std::cos( m*theta + n*zeta );
    double sinmn = std::sin( m*theta + n*zeta );
    double rmnc_i = (*Rmnc_[i])(s); 
    double zmns_i = (*Zmns_[i])(s);
    double d_rmnc_i = (*Rmnc_[i]).derivative(s); 
    double d_zmns_i = (*Zmns_[i]).derivative(s); 
    double d2_rmnc_i = (*Rmnc_[i]).derivative2(s);
    double d2_zmns_i = (*Zmns_[i]).derivative2(s);
    // assuming for now that vmec equilibrium has stellarator symmetry.
    R += rmnc_i * cosmn; Z += zmns_i * sinmn;
    dR_ds += d_rmnc_i * cosmn; 
    dR_dtheta -= m * rmnc_i * sinmn; 
    dR_dzeta -= n * rmnc_i * sinmn; 
    d2R_ds2 += d2_rmnc_i * cosmn; 
    d2R_dsdtheta -= m * d_rmnc_i * sinmn;
    d2R_dsdzeta -= n * d_rmnc_i * sinmn;  
    d2R_dtheta2 -= m * m * rmnc_i * cosmn;
    d2R_dthetadzeta -= m * n * rmnc_i * cosmn;
    d2R_dzeta2 -= n * n * rmnc_i * cosmn;
    dZ_ds += d_zmns_i * sinmn; 
    dZ_dtheta += m * zmns_i * cosmn; 
    dZ_dzeta += n * zmns_i * cosmn;
    d2Z_ds2 += d2_zmns_i * sinmn;
    d2Z_dsdtheta += m * d_zmns_i * cosmn;
    d2Z_dsdzeta += n * d_zmns_i * cosmn;
    d2Z_dtheta2 -= m * m * zmns_i * sinmn;
    d2Z_dthetadzeta -= n * m * zmns_i * sinmn;
    d2Z_dzeta2 -= n * n * zmns_i * sinmn;
}
dSM3 value = {
      2 * (dR_ds * d2R_ds2      + dZ_ds * d2Z_ds2), 
      2 * (dR_ds * d2R_dsdtheta + dZ_ds * d2Z_dsdtheta),  // d_i g_uu
      2 * (dR_ds * d2R_dsdzeta  + dZ_ds * d2Z_dsdzeta),
      dR_ds * d2R_dsdtheta      + dR_dtheta * d2R_ds2      + dZ_ds * d2Z_dsdtheta     + dZ_dtheta * d2Z_ds2,
      dR_ds * d2R_dtheta2       + dR_dtheta * d2R_dsdtheta + dZ_ds * d2Z_dtheta2      + dZ_dtheta * d2Z_dsdtheta, // d_i g_uv
      dR_ds * d2R_dthetadzeta   + dR_dtheta * d2R_dsdzeta  + dZ_ds * d2Z_dthetadzeta  + dZ_dtheta * d2Z_dsdzeta, 
      dR_ds * d2R_dsdzeta       + dR_dzeta * d2R_ds2       + dZ_ds * d2Z_dsdzeta      + dZ_dzeta * d2Z_ds2,
      dR_ds * d2R_dthetadzeta   + dR_dzeta * d2R_dsdtheta  + dZ_ds * d2Z_dthetadzeta  + dZ_dzeta * d2Z_dsdtheta,  // d_i g_uw
      dR_ds * d2R_dzeta2        + dR_dzeta * d2R_dsdzeta   + dZ_ds * d2Z_dzeta2       + dZ_dzeta * d2Z_dsdzeta,
      2 * (dR_dtheta * d2R_dsdtheta     + dZ_dtheta * d2Z_dsdtheta), 
      2 * (dR_dtheta * d2R_dtheta2      + dZ_dtheta * d2Z_dtheta2), // d_i g_vv
      2 * (dR_dtheta * d2R_dthetadzeta  + dZ_dtheta * d2Z_dthetadzeta), 
      dR_dtheta * d2R_dsdzeta     + dR_dzeta * d2R_dsdtheta     + dZ_dtheta * d2Z_dsdzeta      + dZ_dzeta * d2Z_dsdtheta,
      dR_dtheta * d2R_dthetadzeta + dR_dzeta * d2R_dtheta2      + dZ_dtheta * d2Z_dthetadzeta  + dZ_dzeta * d2Z_dtheta2,    // d_i g_vw
      dR_dtheta * d2R_dzeta2      + dR_dzeta * d2R_dthetadzeta  + dZ_dtheta * d2Z_dzeta2       + dZ_dzeta * d2Z_dthetadzeta,
      2 * (R * dR_ds     + dR_dzeta * d2R_dsdzeta     + dZ_dzeta * d2Z_dsdzeta),
      2 * (R * dR_dtheta + dR_dzeta * d2R_dthetadzeta + dZ_dzeta * d2Z_dthetadzeta), // d_i g_ww
      2 * (R * dR_dzeta  + dR_dzeta * d2R_dzeta2      + dZ_dzeta * d2Z_dzeta2),
  };
  if (use_cache_) {
    cached_del_position_ = position;
    cached_del_value_ = value;
  }
  return value;
}
auto metric_vmec::transform2cylindrical(const IR3& position) const -> IR3 {
    double u = position[gyronimo::IR3::u];
    double v = position[gyronimo::IR3::v];
    double w = position[gyronimo::IR3::w];
    if (u < 0.0) {
      u = -u;
      v = v + std::numbers::pi; // @todo, discuss clockwise versus anti-clockwise rotating of theta when mirroring @ s=0
    }
    else if (u > 1) u = 1;
    double R = 0.0, Z = 0.0;
  
    #pragma omp parallel for reduction(+: R, Z)
    for (size_t i = 0; i<xm_.size(); i++) {
      double m = xm_[i]; double n = xn_[i];
      R+= (*Rmnc_[i])(u) * std::cos( m*v + n*w ); 
      Z+= (*Zmns_[i])(u) * std::sin( m*v + n*w );
    }
    return  {R, w, Z};
}
auto metric_vmec::jacobian(const IR3& position) const  -> double {
  double s = position[IR3::u];
  double theta = position[IR3::v];
  double zeta = position[IR3::w];
  if (s < 0.0) {
      s = -s;
      theta = theta + std::numbers::pi; // @todo, discuss clockwise versus anti-clockwise rotating of theta when mirroring @ s=0
  }
  else if (s > (1-ds_half_cell_)) s = (1-ds_half_cell_);
  double J = 0.0;
  #pragma omp parallel for reduction(+: J)
  for (size_t i = 0; i < xm_nyq_.size(); i++) {  
    J += (*gmnc_[i])(s) * std::cos( xm_nyq_[i]*theta + xn_nyq_[i]*zeta );
  };
  return J;
}
} // end namespace gyronimo
