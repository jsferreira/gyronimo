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

// @vmectrace.cc, this file is part of ::gyronimo::

// Command-line tool to print guiding-centre orbits in `VMEC` equilibria.
// External dependencies:
// - [argh](https://github.com/adishavit/argh), a minimalist argument handler.
// - [GSL](https://www.gnu.org/software/gsl), the GNU Scientific Library.
// - [boost](https://www.gnu.org/software/gsl), the boost library.

#ifdef OPENMP
#include <omp.h>
#endif
#include <cmath>
#include <argh.h>
#include <iostream>
#include <gyronimo/version.hh>
#include <gyronimo/core/codata.hh>
#include <gyronimo/core/linspace.hh>
#include <gyronimo/parsers/parser_vmec.hh>
#include <gyronimo/fields/equilibrium_vmec.hh>
#include <gyronimo/metrics/morphism_vmec.hh>
#include <gyronimo/interpolators/cubic_gsl.hh>
#include <gyronimo/dynamics/guiding_centre.hh>
#include <gyronimo/dynamics/odeint_adapter.hh>
#include <boost/math/tools/roots.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta4.hpp>
#include <boost/numeric/odeint/integrate/integrate_const.hpp>


void print_help() {
  std::cout << "vmectrace, powered by gyronimo-v"
      << gyronimo::version_major << "." << gyronimo::version_minor << ".\n";
  std::cout <<
      "usage: vmectrace [options] vmap\n";
  std::cout <<
      "reads an VMEC vmap and prints the required orbit to stdout.\n";
  std::cout << "options:\n";
  std::cout << "  -rhom=rrr      Axis density (in m_proton*1e19, default 1).\n";
  std::cout << "  -mass=mmm      Particle mass (in m_proton, default 1).\n";
  std::cout << "  -charge=qqq    Particle charge (in q_proton, default 1).\n";
  std::cout << "  -s=sss         s = PSIphi/PSIphi_bnd value (normalized, 0 to 1, default 0.5).\n";
  std::cout << "  -theta=ooo     theta = VMEC poloidal angle (normalized, 0 to 1, in 2Pi*rad, default 0.0).\n";
  std::cout << "  -phi=ppp       phi = VMEC/cylindrical toroidal angle (normalized, 0 to 1, in 2Pi*rad, default 0.0).\n";
  std::cout << "  -energy=eee    Energy value (in eV, default 1).\n";
  std::cout << "  -lambda=lll    Lambda value, signed as Vpp (default +1).\n";
  std::cout << "  -tfinal=ttt    Time limit (in R0/Valfven, default 1).\n";
  std::cout << "  -nsamples=nnn  Number of orbit samples (default 512).\n";
  std::cout << "  -cyl_pitch     optionally, use cylindrical coordinates and pitch angle to initialize particle .\n";
  std::cout << "  -R=rrr         Cylindrical coordinate R (in m, default 1.0)\n";
  std::cout << "  -z=rrr         Cylindrical coordinate z (in m, default 0.0)\n";
  std::cout << "  -pitch=ccc     pitch-angle cosine (v.b/|v|).\n";
  std::exit(0);
}

// ODEInt observer object to print diagnostics at each time step.
class orbit_observer {
public:
  orbit_observer(
      // double zstar, 
      double vstar,
      const gyronimo::equilibrium_vmec* e, const gyronimo::morphism_vmec* T, const gyronimo::guiding_centre* g)
      // : zstar_(zstar), 
        : vstar_(vstar), eq_pointer_(e), T_pointer_(T), gc_pointer_(g) {};
  void operator()(const gyronimo::guiding_centre::state& s, double t) {
    gyronimo::IR3 q = gc_pointer_->get_position(s);
    double v_parallel = gc_pointer_->get_vpp(s);
//    double bphi = eq_pointer_->covariant_versor(x, t)[gyronimo::IR3::w];
//    double flux = x[gyronimo::IR3::u]*x[gyronimo::IR3::u];
    auto X = (*T_pointer_)(q);
    double R = std::sqrt(X[gyronimo::IR3::u]*X[gyronimo::IR3::u]
                       + X[gyronimo::IR3::v]*X[gyronimo::IR3::v]);
//    gyronimo::IR3 X = eq_pointer_->metric()->transform2cylindrical(x);
    std::cout << t << " "
        << q[gyronimo::IR3::u] << " "
        << q[gyronimo::IR3::v] << " "
        << q[gyronimo::IR3::w] << " "
        << v_parallel << " "
        // << -zstar_*flux + vstar_*v_parallel*bphi << " "
        << gc_pointer_->energy_perpendicular(s, t) << " "
        << gc_pointer_->energy_parallel(s) << " " 
        << R << " "                     // R
        << q[gyronimo::IR3::u] << " "   // phi
        << X[gyronimo::IR3::w] << " "   // z
        << X[gyronimo::IR3::u] << " "   // x
        << X[gyronimo::IR3::v] << " "   // y
        << X[gyronimo::IR3::w] << "\n"; // z
  };
private:
  // double zstar_;
  double vstar_;
  const gyronimo::equilibrium_vmec* eq_pointer_;
  const gyronimo::morphism_vmec* T_pointer_;
  const gyronimo::guiding_centre* gc_pointer_;
};

int main(int argc, char* argv[]) {
  auto command_line = argh::parser(argv);
  if (command_line[{"h", "help"}]) print_help();
  if (!command_line(1)) {  // the 1st non-option argument is the mapping file.
    std::cout << "vmectrace: no VMEC equilibrium file provided; -h for help.\n";
    std::exit(1);
  }
  gyronimo::parser_vmec vmap(command_line[1]);
  gyronimo::cubic_gsl_factory ifactory;
  gyronimo::metric_vmec g(&vmap, &ifactory);
  gyronimo::morphism_vmec T(&vmap, &ifactory);
  gyronimo::equilibrium_vmec veq(&g, &ifactory);

// Reads parameters from the command line:
//  double pphi; command_line("pphi", 1.0) >> pphi;  // pphi in eV.s.
  double s; command_line("s", 0.5) >> s;  // s, normalized
  double theta; command_line("theta", 0.0) >> theta; theta *= 2.0*M_PI;  // theta in radians;
  double phi; command_line("phi", 0.0) >> phi; phi *= 2.0*M_PI; // phi in radians;
  double mass; command_line("mass", 1.0) >> mass;  // m_proton units.
  double rhom; command_line("rhom", 1.0) >> rhom;  // density in m_proton*1e19.
  double charge; command_line("charge", 1.0) >> charge;  // q_electron units.
  double energy; command_line("energy", 1.0) >> energy;  // energy in eV.
  double lambda; command_line("lambda", 1.0) >> lambda;  // lambda is signed!
  double Tfinal; command_line("tfinal", 1.0) >> Tfinal;
  double pitch; command_line("pitch", 1.0) >> pitch;     // velocity pitch cosine
  double R; command_line("R", 1.0) >> R;          // R coordinate for Cylindrical coordinates
  double z; command_line("z", 1.0) >> z;          // z coordinate for Cylindrical coordinates

// Computes normalisation constants:
  double Valfven = veq.B_0()/std::sqrt(
      gyronimo::codata::mu0*(rhom*gyronimo::codata::m_proton*1.e+19));
  double Ualfven = 0.5*gyronimo::codata::m_proton*mass*Valfven*Valfven;
  double energySI = energy*gyronimo::codata::e;
  double Lref = veq.R_0();
// Initialize particle:
  double v2 = 2.0*energySI/(gyronimo::codata::m_proton*mass);
  double v = std::sqrt(v2);
  double vpr, vpp, vpp_sign;
  if (command_line["cyl_pitch"]) {  // if initialize in cylindrical coordinates
    gyronimo::IR3 X {R*std::cos(phi), R*std::sin(phi), z};
    auto q_ = T.inverse(X);
    vpp = pitch * v; vpr = std::sqrt(v2-vpp*vpp);
    vpp_sign = std::copysign(1.0, vpp);  // pitch carries vpp sign.
    double BoB0 = veq.magnitude(q_, 0.0);
    lambda = (1.0 - pitch*pitch)/BoB0;
    s = q_[gyronimo::IR3::u]; theta = q_[gyronimo::IR3::w]; //phi = q_[gyronimo::IR3::v],
  } 
  else {
    gyronimo::IR3 q_ {s, phi, theta};
    double B = veq.magnitude(q_, 0.0);
    vpp_sign = std::copysign(1.0, lambda);  // lambda carries vpp sign.
    lambda = std::abs(lambda);  // once vpp sign is stored, lambda turns unsigned.
    double vpr2 =  2.0*energySI*B*lambda/(gyronimo::codata::m_proton*mass*veq.B_0());
    vpp = std::sqrt(v2-vpr2); vpr = std::sqrt(vpr2); 
    lambda = std::abs(lambda);  
  };
  gyronimo::IR3 q {s, phi, theta};
// Prints output header:
  std::cout << "# vmectrace, powered by ::gyronimo:: v"
      << gyronimo::version_major << "." << gyronimo::version_minor << ".\n";
  std::cout << "# args: ";
  for(int i = 1; i < argc; i++) std::cout << argv[i] << " ";
  std::cout << std::endl;
  std::cout << "# l_ref = " << Lref << " [m];";
  std::cout << " v_alfven = " << Valfven << " [m/s];";
  std::cout << " u_alfven = " << Ualfven << " [J];";
  std::cout << " energy = " << energySI << " [J], lambda = " << lambda << " [-].\n";
  std::cout << " s = " << q[gyronimo::IR3::u] << ", phi = " << q[gyronimo::IR3::v] << ", theta = " << q[gyronimo::IR3::w] << "\n";
  std::cout << " |v| = " << v<< " [m/s], v_par = " << vpp << " [m/s], v_per = " << vpr << " [m/s].\n";
  std::cout << "#\n";
  std::cout << "# vars: t s theta zeta vpar Pphi/e Eperp/Ealfven Epar/Ealfven R phi Z x y z\n";

// Builds the guiding_centre object:
  gyronimo::guiding_centre gc(
      Lref, Valfven, charge/mass, lambda*energySI/Ualfven, &veq);

// Computes the initial conditions from the supplied constants of motion:
  // double zstar = charge*g.parser()->cpsurf()*veq.B_0()*veq.R_0()*veq.R_0();
  double vstar = Valfven*mass*gyronimo::codata::m_proton/gyronimo::codata::e;
  double vdagger = vstar*std::sqrt(energySI/Ualfven);
  
  gyronimo::guiding_centre::state initial_state = gc.generate_state(
      q, energySI/Ualfven,
      (vpp_sign > 0 ? gyronimo::guiding_centre::plus : gyronimo::guiding_centre::minus));

// integrates for t in [0,Tfinal], with dt=Tfinal/nsamples, using RK4.
  std::cout.precision(16);
  std::cout.setf(std::ios::scientific);
  orbit_observer observer(vstar, &veq, &T, &gc);
  std::size_t nsamples; command_line("nsamples", 512) >> nsamples;
  boost::numeric::odeint::runge_kutta4<gyronimo::guiding_centre::state>
      integration_algorithm;
  boost::numeric::odeint::integrate_const(
      integration_algorithm, gyronimo::odeint_adapter(&gc),
      initial_state, 0.0, Tfinal, Tfinal/nsamples, observer);
  return 0;
}
