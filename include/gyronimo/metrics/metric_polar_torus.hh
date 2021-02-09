// ::gyronimo:: - gyromotion for the people, by the people -
// An object-oriented library for gyromotion applications in plasma physics.
// Copyright (C) 2021 Paulo Rodrigues.

// @metric_polar_torus.hh

#ifndef GYRONIMO_METRIC_POLAR_TORUS
#define GYRONIMO_METRIC_POLAR_TORUS

#include <gyronimo/metrics/metric_covariant.hh>

namespace gyronimo {

//! Covariant metric for toroidal coordinates with polar cross section.
/*!
    The three contravariant coordinates are the distance to the magnetic axis
    normalized to the `minor_radius` (`u`), the angle measured counterclockwise
    on the poloidal cross section from the low-field side midplane (`v`, in
    rads), and the toroidal angle (`w`, in rads) measured clockwise when looking
    from the torus' top. The lengths `minor_radius` and `major_radius` are in SI
    units. Inherited methods `Jacobian`, `to_covariant`, and `to_contravariant`
    are overriden for efficiency.
*/
class metric_polar_torus : public metric_covariant {
 public:
  metric_polar_torus(const double minor_radius, double major_radius);
  virtual ~metric_polar_torus() override {};

  virtual SM3 operator()(const IR3& r) const override;
  virtual dSM3 del(const IR3& r) const override;

  virtual double jacobian(const IR3& r) const override;
  virtual IR3 to_covariant(const IR3& B, const IR3& r) const override;
  virtual IR3 to_contravariant(const IR3& B, const IR3& r) const override;

  double minor_radius() const {return minor_radius_;};
  double major_radius() const {return major_radius_;};
  double iaspect_ratio() const {return iaspect_ratio_;};

 private:
  const double minor_radius_, major_radius_;
  const double minor_radius_squared_, major_radius_squared_, iaspect_ratio_;
};

} // end namespace gyronimo.

#endif // GYRONIMO_METRIC_POLAR_TORUS
